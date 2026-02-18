import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import json
import os
from datetime import datetime

# ================== CONFIGURACIÓN ==================
st.set_page_config(layout="wide", page_title="APEX 150K ELITE")

# Archivos
PORTFOLIO_FILE = "portfolio.json"

# Parámetros fijos
TARGET_GOAL = 150000
STRUCTURAL_RESERVE_PCT = 0.08
SECTOR_CAP = 0.35          # Límite por sector (ej. semis + uranio juntos)
DEFAULT_MONTHLY = 400       # Aporte mensual por defecto

# Lista de tickers (orden fijo)
TICKERS = ["BTC-EUR", "EMXC.DE", "IS3Q.DE", "PPFB.DE", "U3O8.DE", "VVSM.DE", "ZPRR.DE"]

# Mapeo sectorial
SECTOR_MAP = {
    "BTC-EUR": "crypto",
    "EMXC.DE": "equity_em",
    "IS3Q.DE": "equity_em",
    "PPFB.DE": "commodity",
    "U3O8.DE": "commodity",
    "VVSM.DE": "semis",
    "ZPRR.DE": "real_estate"
}

# ================== FUNCIONES DE PERSISTENCIA ==================
def load_portfolio():
    """Carga cartera desde JSON. Si no existe, crea una por defecto."""
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            data = json.load(f)
        # Asegurar que todas las claves existen
        if "positions" not in data:
            data["positions"] = {}
        for t in TICKERS:
            if t not in data["positions"]:
                data["positions"][t] = {"shares": 0, "avg_price": 0}
        if "cash_reserve" not in data:
            data["cash_reserve"] = 0
        return data
    else:
        # Estructura por defecto
        default = {
            "positions": {t: {"shares": 0, "avg_price": 0} for t in TICKERS},
            "cash_reserve": 0,
            "last_updated": datetime.now().isoformat()
        }
        save_portfolio(default)
        return default

def save_portfolio(portfolio):
    """Guarda cartera en JSON."""
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)

# ================== DATOS DE MERCADO ==================
@st.cache_data(ttl=300)
def get_market_data():
    """Descarga precios y datos macro (5 años)."""
    all_tickers = TICKERS + ["^VIX", "^TNX", "^GSPC"]
    try:
        raw = yf.download(all_tickers, period="5y", auto_adjust=True, progress=False)["Close"]
    except:
        raw = yf.download(all_tickers, period="5y", auto_adjust=True, progress=False)["Close"]
    raw = raw.ffill()
    prices = raw[TICKERS]
    macro = raw[["^VIX", "^TNX", "^GSPC"]]
    return prices, macro

# ================== RÉGIMEN DE MERCADO ==================
def get_regime(vix, vix_series):
    vix_p80 = vix_series.quantile(0.8)
    vix_p20 = vix_series.quantile(0.2)
    if vix > vix_p80:
        return "RISK_OFF", 0.10
    elif vix < vix_p20:
        return "RISK_ON", 0.18
    else:
        return "NEUTRAL", 0.14

def check_btc_attack(btc_series):
    ma200 = btc_series.rolling(200).mean()
    std200 = btc_series.rolling(200).std()
    btc_z = (btc_series.iloc[-1] - ma200.iloc[-1]) / std200.iloc[-1]
    return btc_z < -2, btc_z

# ================== OPTIMIZACIÓN CON LÍMITES EN BTC ==================
def optimize_portfolio(returns, target_vol, btc_min, btc_max, sector_map, sector_cap):
    """
    Maximiza Sharpe con restricciones:
      - volatilidad <= target_vol
      - límites individuales (btc entre btc_min y btc_max, otros entre 2% y 40%)
      - límite por sector (sector_cap)
    """
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    n = len(returns.columns)
    
    def neg_sharpe(w):
        port_return = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        return -port_return / port_vol
    
    # Restricciones
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    constraints.append({'type': 'ineq', 'fun': lambda w: target_vol - np.sqrt(w @ cov @ w)})
    
    # Límites sectoriales
    for sector in set(sector_map.values()):
        indices = [i for i, t in enumerate(returns.columns) if sector_map[t] == sector]
        if indices:
            constraints.append({'type': 'ineq', 'fun': lambda w, idx=indices: sector_cap - np.sum(w[idx])})
    
    # Límites individuales
    bounds = [(0.02, 0.40) for _ in range(n)]
    btc_idx = returns.columns.get_loc("BTC-EUR")
    bounds[btc_idx] = (btc_min, btc_max)   # Forzamos el rango de BTC
    
    # Punto inicial
    w0 = np.ones(n) / n
    result = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints,
                      method='SLSQP', options={'ftol': 1e-6})
    
    if not result.success:
        # Fallback: mínima varianza
        def port_vol(w): return np.sqrt(w @ cov @ w)
        result = minimize(port_vol, w0, bounds=bounds, constraints=constraints, method='SLSQP')
    
    return pd.Series(result.x, index=returns.columns)

# ================== CONTRIBUCIÓN AL RIESGO ==================
def risk_contribution(weights, cov):
    port_var = weights @ cov @ weights
    marginal_contrib = cov @ weights
    risk_contrib = weights * marginal_contrib / np.sqrt(port_var)
    return risk_contrib / risk_contrib.sum()

# ================== MONTE CARLO ==================
def run_monte_carlo(current_value, monthly_injection, years, mu, vol, n_sims=5000):
    months = years * 12
    monthly_mu = mu / 12
    monthly_vol = vol / np.sqrt(12)
    results = []
    for _ in range(n_sims):
        value = current_value
        for m in range(months):
            ret = np.random.normal(monthly_mu, monthly_vol)
            value = value * (1 + ret) + monthly_injection
        results.append(value)
    return np.array(results)

# ================== GENERAR ÓRDENES ==================
def generate_orders(current_weights, target_weights, current_values, cash_available, prices):
    total_value = sum(current_values.values())
    target_values = {t: target_weights[t] * (total_value + cash_available) for t in target_weights.index}
    orders = {}
    spent = 0
    for t in target_weights.index:
        current = current_values.get(t, 0)
        target = target_values[t]
        diff = target - current
        if diff > 0:
            price = prices[t]
            if t == "BTC-EUR":
                units = round(diff / price, 6)
                if units * price <= cash_available - spent:
                    orders[t] = units
                    spent += units * price
            else:
                units = int(diff // price)
                if units > 0 and units * price <= cash_available - spent:
                    orders[t] = units
                    spent += units * price
    return orders, spent

def execute_orders(portfolio, orders, prices):
    for t, units in orders.items():
        price = prices[t]
        old = portfolio["positions"][t]
        new_shares = old["shares"] + units
        new_avg = (old["avg_price"] * old["shares"] + units * price) / new_shares if new_shares > 0 else 0
        portfolio["positions"][t] = {"shares": new_shares, "avg_price": new_avg}
        portfolio["cash_reserve"] -= units * price
    portfolio["last_updated"] = datetime.now().isoformat()
    save_portfolio(portfolio)
    return portfolio

# ================== INTERFAZ PRINCIPAL ==================
def main():
    st.title("🦅 **APEX 150K ELITE** — HEDGE FUND EDITION")
    
    # Cargar cartera
    portfolio = load_portfolio()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controles")
        monthly_injection = st.number_input("Aporte mensual (€)", min_value=0, value=DEFAULT_MONTHLY, step=50)
        btc_min = st.slider("Peso mínimo BTC", min_value=0.0, max_value=0.25, value=0.20, step=0.01, format="%.2f")
        btc_max = st.slider("Peso máximo BTC", min_value=btc_min, max_value=0.40, value=0.25, step=0.01, format="%.2f")
        
        st.markdown("---")
        st.subheader("💾 Estado cartera")
        st.json({
            "Reserva actual": f"{portfolio['cash_reserve']:.2f} €",
            "Última actualización": portfolio.get("last_updated", "N/A")
        })
        if st.button("⟳ Recargar datos"):
            st.cache_data.clear()
            st.rerun()
    
    # Obtener datos de mercado
    prices_df, macro_df = get_market_data()
    latest_prices = prices_df.iloc[-1]
    
    # Calcular valor actual
    current_values = {}
    for t in TICKERS:
        shares = portfolio["positions"].get(t, {}).get("shares", 0)
        current_values[t] = shares * latest_prices[t]
    current_total = sum(current_values.values())
    current_weights = pd.Series({t: current_values[t]/current_total for t in TICKERS})
    
    # Datos macro y régimen
    vix = macro_df["^VIX"].iloc[-1]
    vix_series = macro_df["^VIX"]
    regime, target_vol = get_regime(vix, vix_series)
    
    # Ataque BTC
    btc_series = prices_df["BTC-EUR"]
    attack_mode, btc_z = check_btc_attack(btc_series)
    if attack_mode:
        regime = "ATTACK_MODE"
        target_vol = 0.22  # Más riesgo en ataque
    
    # Optimizar pesos objetivo con los límites elegidos
    returns = prices_df.pct_change().dropna()
    target_weights = optimize_portfolio(returns, target_vol, btc_min, btc_max, SECTOR_MAP, SECTOR_CAP)
    
    # Contribución al riesgo actual
    cov = returns.cov() * 252
    risk_contrib = risk_contribution(current_weights.values, cov)
    
    # Disponible para invertir
    total_cash = portfolio["cash_reserve"] + monthly_injection
    structural_reserve = STRUCTURAL_RESERVE_PCT * (current_total + monthly_injection)
    usable_cash = max(0, total_cash - structural_reserve)
    if attack_mode:
        usable_cash = total_cash  # En ataque podemos usar todo (opcional)
    
    # Generar órdenes
    orders, spent = generate_orders(current_weights, target_weights, current_values, usable_cash, latest_prices)
    remaining_cash = total_cash - spent
    
    # ================== DASHBOARD ==================
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RÉGIMEN", regime, delta=f"VIX {vix:.1f}")
    col2.metric("BTC Precio", f"{latest_prices['BTC-EUR']:,.0f} €", delta=f"Z-score {btc_z:.2f}")
    
    # Probabilidad (escenario base)
    expected_return = target_weights @ (returns.mean() * 252)
    mc_base = run_monte_carlo(current_total, monthly_injection, 10, expected_return, target_vol)
    prob_base = np.mean(mc_base >= TARGET_GOAL)
    col3.metric("Probabilidad 150K", f"{prob_base:.1%}")
    col4.metric("Reserva actual", f"{portfolio['cash_reserve']:.2f} €", delta=f"Disponible: {usable_cash:.2f}")
    
    st.divider()
    
    # ================== GAUGES MACRO ==================
    st.subheader("📊 Panorama Macro")
    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    
    # Gauge VIX
    vix_p80 = vix_series.quantile(0.8)
    fig_vix = go.Figure(go.Indicator(
        mode="gauge+number", value=vix, title="VIX",
        gauge={'axis': {'range': [0, 40]},
               'bar': {'color': 'darkblue'},
               'steps': [{'range': [0, 20], 'color': 'lightgreen'},
                         {'range': [20, 30], 'color': 'yellow'},
                         {'range': [30, 40], 'color': 'red'}],
               'threshold': {'line': {'color': 'black', 'width': 4}, 'thickness': 0.75, 'value': vix_p80}}))
    fig_vix.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
    col_g1.plotly_chart(fig_vix, use_container_width=True)
    
    # Gauge ERP (aproximado)
    risk_free = macro_df["^TNX"].iloc[-1] / 100 if not pd.isna(macro_df["^TNX"].iloc[-1]) else 0.04
    earnings_yield = 0.045  # Placeholder (mejorable con datos reales)
    erp = earnings_yield - risk_free
    fig_erp = go.Figure(go.Indicator(
        mode="gauge+number", value=erp*100, title="ERP (earning yield - 10y)",
        number={'suffix': '%'},
        gauge={'axis': {'range': [-2, 6]},
               'bar': {'color': 'darkgreen'},
               'steps': [{'range': [-2, 1], 'color': 'red'},
                         {'range': [1, 3], 'color': 'yellow'},
                         {'range': [3, 6], 'color': 'lightgreen'}],
               'threshold': {'line': {'color': 'black', 'width': 4}, 'thickness': 0.75, 'value': 2.5}}))
    fig_erp.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
    col_g2.plotly_chart(fig_erp, use_container_width=True)
    
    # Gauge Drawdown S&P
    sp500 = macro_df["^GSPC"]
    rolling_max = sp500.expanding().max()
    drawdown = (sp500 - rolling_max) / rolling_max * 100
    current_dd = drawdown.iloc[-1]
    fig_dd = go.Figure(go.Indicator(
        mode="gauge+number", value=current_dd, title="S&P 500 Drawdown %",
        number={'suffix': '%'},
        gauge={'axis': {'range': [-50, 0]},
               'bar': {'color': 'darkred'},
               'steps': [{'range': [-10, 0], 'color': 'lightgreen'},
                         {'range': [-20, -10], 'color': 'yellow'},
                         {'range': [-50, -20], 'color': 'red'}],
               'threshold': {'line': {'color': 'black', 'width': 4}, 'thickness': 0.75, 'value': -15}}))
    fig_dd.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
    col_g3.plotly_chart(fig_dd, use_container_width=True)
    
    # Gauge Z-score BTC
    fig_z = go.Figure(go.Indicator(
        mode="gauge+number", value=btc_z, title="BTC Z-score (200d)",
        gauge={'axis': {'range': [-3, 3]},
               'bar': {'color': 'orange'},
               'steps': [{'range': [-1, 1], 'color': 'lightgreen'},
                         {'range': [1, 2], 'color': 'yellow'},
                         {'range': [2, 3], 'color': 'red'},
                         {'range': [-2, -1], 'color': 'yellow'},
                         {'range': [-3, -2], 'color': 'red'}],
               'threshold': {'line': {'color': 'black', 'width': 4}, 'thickness': 0.75, 'value': -2}}))
    fig_z.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
    col_g4.plotly_chart(fig_z, use_container_width=True)
    
    st.divider()
    
    # ================== DONUTS ==================
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.subheader("🎯 Asignación Objetivo")
        fig_target = px.pie(names=target_weights.index, values=target_weights.values, hole=0.6)
        st.plotly_chart(fig_target, use_container_width=True)
    with col_d2:
        st.subheader("📉 Contribución al Riesgo Actual")
        risk_df = pd.DataFrame({"Activo": target_weights.index, "Contribución": risk_contrib})
        fig_risk = px.pie(risk_df, names="Activo", values="Contribución", hole=0.6)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Tabla desviaciones
    st.subheader("📋 Desviación vs Objetivo")
    df_compare = pd.DataFrame({
        "Objetivo": target_weights,
        "Actual": current_weights,
        "Diferencia": current_weights - target_weights,
        "Valor actual (€)": [current_values[t] for t in target_weights.index],
        "Precio": [latest_prices[t] for t in target_weights.index]
    })
    st.dataframe(df_compare.style.format({
        "Objetivo": "{:.2%}",
        "Actual": "{:.2%}",
        "Diferencia": "{:.2%}",
        "Valor actual (€)": "{:.2f}",
        "Precio": "{:.2f}"
    }))
    
    st.divider()
    
    # ================== MONTE CARLO ESCENARIOS ==================
    st.subheader("📈 Monte Carlo 10 años (escenarios)")
    mu_base = expected_return
    vol_base = target_vol
    mu_conserv = mu_base - 0.02
    vol_conserv = vol_base + 0.02
    mu_opt = mu_base + 0.02
    vol_opt = vol_base - 0.02
    
    mc_conserv = run_monte_carlo(current_total, monthly_injection, 10, mu_conserv, vol_conserv)
    mc_base = run_monte_carlo(current_total, monthly_injection, 10, mu_base, vol_base)
    mc_opt = run_monte_carlo(current_total, monthly_injection, 10, mu_opt, vol_opt)
    
    prob_conserv = np.mean(mc_conserv >= TARGET_GOAL)
    prob_base = np.mean(mc_base >= TARGET_GOAL)
    prob_opt = np.mean(mc_opt >= TARGET_GOAL)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Conservador", f"{prob_conserv:.1%}")
    col_m2.metric("Base", f"{prob_base:.1%}")
    col_m3.metric("Optimista", f"{prob_opt:.1%}")
    
    # Histograma
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(x=mc_conserv, name="Conservador", opacity=0.5))
    fig_mc.add_trace(go.Histogram(x=mc_base, name="Base", opacity=0.5))
    fig_mc.add_trace(go.Histogram(x=mc_opt, name="Optimista", opacity=0.5))
    fig_mc.add_vline(x=TARGET_GOAL, line_dash="dash", line_color="red", annotation_text="150K")
    fig_mc.update_layout(barmode='overlay', title="Distribución de valor final")
    st.plotly_chart(fig_mc, use_container_width=True)
    
    st.divider()
    
    # ================== ÓRDENES ==================
    st.subheader("🛒 Órdenes sugeridas")
    if orders:
        for t, units in orders.items():
            cost = units * latest_prices[t]
            st.write(f"• **{t}**: comprar {units} unidades a {latest_prices[t]:.2f} € → coste {cost:.2f} €")
        st.write(f"**Coste total:** {spent:.2f} €")
        st.write(f"**Reserva restante tras compras:** {remaining_cash:.2f} €")
        
        if st.button("✅ Confirmar ejecución"):
            portfolio = execute_orders(portfolio, orders, latest_prices)
            st.success("Órdenes ejecutadas. Cartera actualizada.")
            st.rerun()
    else:
        st.info("No hay órdenes generadas (saldo insuficiente o cartera ya equilibrada).")
    
    st.write(f"**Reserva estructural recomendada (8%):** {structural_reserve:.2f} €")
    st.write(f"**Reserva real tras operación:** {remaining_cash:.2f} €")
    
    st.divider()
    
    # ================== DIAGNÓSTICO ==================
    st.subheader("🧠 Diagnóstico de Mercado")
    if regime == "RISK_ON":
        st.success("🔵 Régimen RISK ON: volatilidad baja. Momento favorable.")
    elif regime == "RISK_OFF":
        st.warning("🟠 Régimen RISK OFF: volatilidad alta. Priorizando defensa.")
    elif regime == "ATTACK_MODE":
        st.error("🔴 MODO ATAQUE: BTC en capitulación extrema. Aumentando exposición táctica.")
    else:
        st.info("⚪ Régimen NEUTRAL: posicionamiento equilibrado.")
    
    st.caption(f"Última actualización de precios: {prices_df.index[-1].strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    main()
