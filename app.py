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

PORTFOLIO_FILE = "portfolio.json"
TARGET_GOAL = 150000
STRUCTURAL_RESERVE_PCT = 0.08
DEFAULT_MONTHLY = 400

# Lista de tickers (todos deben coincidir con los del JSON)
TICKERS = ["BTC-EUR", "EMXC.DE", "IS3Q.DE", "PPFB.DE", "URNU.DE", "VVSM.DE", "ZPRR.DE"]

# Mapeo sectorial
SECTOR_MAP = {
    "BTC-EUR": "crypto",
    "EMXC.DE": "emerging",
    "IS3Q.DE": "global_quality",
    "PPFB.DE": "gold",
    "URNU.DE": "uranium",
    "VVSM.DE": "semis",
    "ZPRR.DE": "smallcap_usa"
}

SECTOR_CAP = 0.35  # Límite por sector

# ================== FUNCIONES DE PERSISTENCIA CON EDITOR ==================
def load_portfolio():
    """Carga la cartera desde JSON. Si hay error, permite editarlo en la interfaz."""
    # Datos por defecto (tus posiciones)
    default_positions = {
        "BTC-EUR": {"shares": 0.031285, "avg_price": 88010.99},
        "EMXC.DE": {"shares": 31, "avg_price": 28.93},
        "IS3Q.DE": {"shares": 26, "avg_price": 67.53},
        "PPFB.DE": {"shares": 4, "avg_price": 69.39},
        "URNU.DE": {"shares": 13, "avg_price": 26.48},
        "VVSM.DE": {"shares": 2, "avg_price": 52.01},
        "ZPRR.DE": {"shares": 6, "avg_price": 61.67}
    }
    default = {
        "positions": default_positions,
        "cash_reserve": 150,
        "last_updated": datetime.now().isoformat()
    }

    # Si el archivo no existe, lo creamos con los datos por defecto
    if not os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2, ensure_ascii=False)
        return default

    # Intentar cargar el archivo
    try:
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        # Mostrar error y permitir edición
        st.error(f"**Error en el archivo `portfolio.json`:** {str(e)}")
        st.markdown("### Edita el contenido y pulsa Guardar")

        # Leer contenido actual (puede tener caracteres raros)
        with open(PORTFOLIO_FILE, "r", encoding="utf-8", errors="ignore") as f:
            raw_content = f.read()

        # Editor de texto
        new_content = st.text_area("Contenido actual (corrígelo si es necesario):", raw_content, height=400)

        if st.button("💾 Guardar y reiniciar"):
            # Validar el nuevo contenido
            try:
                json.loads(new_content)  # Comprobar si es válido
                # Guardar copia de seguridad
                backup_name = f"portfolio_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(backup_name, "w", encoding="utf-8") as f:
                    f.write(raw_content)
                # Guardar nuevo contenido
                with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
                    f.write(new_content)
                st.success("Archivo guardado correctamente. Reiniciando...")
                st.rerun()
            except json.JSONDecodeError as e2:
                st.error(f"El JSON sigue siendo inválido: {e2}. Corrígelo y vuelve a intentar.")
        st.stop()  # No continuar hasta que se solucione

    # Asegurar que todas las claves existen
    for t in TICKERS:
        if t not in data["positions"]:
            data["positions"][t] = {"shares": 0, "avg_price": 0}
    if "cash_reserve" not in data:
        data["cash_reserve"] = 150
    return data

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False)

# ================== DATOS DE MERCADO ==================
@st.cache_data(ttl=300)
def get_market_data():
    """Descarga datos de Yahoo Finance. Si falla, usa datos simulados."""
    all_tickers = TICKERS + ["^VIX", "^TNX", "^GSPC"]
    try:
        raw = yf.download(all_tickers, period="5y", auto_adjust=True, progress=False)["Close"]
        raw = raw.ffill().bfill()
        prices = raw[TICKERS]
        macro = raw[["^VIX", "^TNX", "^GSPC"]]
    except Exception as e:
        st.warning(f"Error al descargar datos: {e}. Usando datos simulados (no reales).")
        # Datos sintéticos
        dates = pd.date_range(end=datetime.now(), periods=252*5, freq='B')
        prices = pd.DataFrame(index=dates, columns=TICKERS)
        for t in TICKERS:
            prices[t] = np.random.randn(len(dates)).cumsum() + 100
        macro = pd.DataFrame(index=dates, columns=["^VIX", "^TNX", "^GSPC"])
        macro["^VIX"] = np.random.uniform(10, 30, len(dates))
        macro["^TNX"] = np.random.uniform(1, 5, len(dates))
        macro["^GSPC"] = np.random.randn(len(dates)).cumsum() + 4000
    return prices, macro

# ================== RESTO DE FUNCIONES (sin cambios) ==================
# (Se mantienen igual que en la versión anterior: get_regime, check_btc_attack,
# optimize_portfolio, risk_contribution, run_monte_carlo, generate_orders, execute_orders)
# Por brevedad, incluyo solo las cabeceras, pero debes copiar el código completo desde la respuesta anterior.
# Asegúrate de que todas las funciones estén presentes.

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

def optimize_portfolio(returns, target_vol, btc_min, btc_max, sector_map, sector_cap):
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    n = len(returns.columns)
    
    def neg_sharpe(w):
        port_return = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        return -port_return / port_vol
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    constraints.append({'type': 'ineq', 'fun': lambda w: target_vol - np.sqrt(w @ cov @ w)})
    
    for sector in set(sector_map.values()):
        indices = [i for i, t in enumerate(returns.columns) if sector_map[t] == sector]
        if indices:
            constraints.append({'type': 'ineq', 'fun': lambda w, idx=indices: sector_cap - np.sum(w[idx])})
    
    bounds = [(0.02, 0.40) for _ in range(n)]
    btc_idx = returns.columns.get_loc("BTC-EUR")
    bounds[btc_idx] = (btc_min, btc_max)
    
    w0 = np.ones(n) / n
    result = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints,
                      method='SLSQP', options={'ftol': 1e-6})
    
    if not result.success:
        def port_vol(w): return np.sqrt(w @ cov @ w)
        result = minimize(port_vol, w0, bounds=bounds, constraints=constraints, method='SLSQP')
    
    return pd.Series(result.x, index=returns.columns)

def risk_contribution(weights, cov):
    port_var = weights @ cov @ weights
    marginal_contrib = cov @ weights
    risk_contrib = weights * marginal_contrib / np.sqrt(port_var)
    return risk_contrib / risk_contrib.sum()

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
    
    portfolio = load_portfolio()
    
    with st.sidebar:
        st.header("⚙️ Controles")
        monthly_injection = st.number_input("Aporte mensual (€)", min_value=0, value=DEFAULT_MONTHLY, step=50)
        btc_min = st.slider("Peso mínimo BTC", min_value=0.0, max_value=0.40, value=0.20, step=0.01, format="%.2f")
        btc_max = st.slider("Peso máximo BTC", min_value=btc_min, max_value=0.40, value=0.30, step=0.01, format="%.2f")
        
        st.markdown("---")
        st.subheader("💾 Estado cartera")
        st.json({
            "Reserva actual": f"{portfolio['cash_reserve']:.2f} €",
            "Última actualización": portfolio.get("last_updated", "N/A")
        })
        if st.button("⟳ Recargar datos"):
            st.cache_data.clear()
            st.rerun()
    
    # Datos de mercado
    prices_df, macro_df = get_market_data()
    latest_prices = prices_df.iloc[-1]
    
    # Calcular valor actual
    current_values = {}
    for t in TICKERS:
        shares = portfolio["positions"].get(t, {}).get("shares", 0)
        current_values[t] = shares * latest_prices[t]
    current_total = sum(current_values.values())
    if current_total == 0:
        st.warning("La cartera actual tiene valor 0. Revisa las posiciones o los precios.")
        current_weights = pd.Series({t: 1/len(TICKERS) for t in TICKERS})
    else:
        current_weights = pd.Series({t: current_values[t]/current_total for t in TICKERS})
    
    # Datos macro
    vix = macro_df["^VIX"].iloc[-1]
    vix_series = macro_df["^VIX"]
    regime, target_vol = get_regime(vix, vix_series)
    
    # Ataque BTC
    btc_series = prices_df["BTC-EUR"]
    attack_mode, btc_z = check_btc_attack(btc_series)
    if attack_mode:
        regime = "ATTACK_MODE"
        target_vol = 0.22
    
    # Optimizar
    returns = prices_df.pct_change().dropna()
    if returns.empty:
        st.error("No hay suficientes datos para calcular rentabilidades. Usando datos simulados.")
        returns = pd.DataFrame(np.random.randn(100, len(TICKERS)), columns=TICKERS) * 0.01
    
    target_weights = optimize_portfolio(returns, target_vol, btc_min, btc_max, SECTOR_MAP, SECTOR_CAP)
    
    # Risk contribution
    cov = returns.cov() * 252
    risk_contrib = risk_contribution(current_weights.values, cov)
    
    # Efectivo disponible
    total_cash = portfolio["cash_reserve"] + monthly_injection
    structural_reserve = STRUCTURAL_RESERVE_PCT * (current_total + monthly_injection)
    usable_cash = total_cash if attack_mode else max(0, total_cash - structural_reserve)
    
    # Órdenes
    orders, spent = generate_orders(current_weights, target_weights, current_values, usable_cash, latest_prices)
    remaining_cash = total_cash - spent
    
    # ================== DASHBOARD ==================
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RÉGIMEN", regime, delta=f"VIX {vix:.1f}")
    col2.metric("BTC Precio", f"{latest_prices['BTC-EUR']:,.0f} €", delta=f"Z-score {btc_z:.2f}")
    
    expected_return = target_weights @ (returns.mean() * 252)
    mc_base = run_monte_carlo(current_total, monthly_injection, 10, expected_return, target_vol)
    prob_base = np.mean(mc_base >= TARGET_GOAL)
    col3.metric("Probabilidad 150K", f"{prob_base:.1%}")
    col4.metric("Reserva actual", f"{portfolio['cash_reserve']:.2f} €", delta=f"Disponible: {usable_cash:.2f}")
    
    st.divider()
    
    # Gauges macro (incluirlos aquí, pero por brevedad no los repito; copiar de la versión anterior)
    # ... (pon los mismos gauges que antes)
    
    st.divider()
    
    # Donuts y tabla
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
    
    # Monte Carlo escenarios
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
    
    # Órdenes
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
    
    st.write(f"**Reserva estructural objetivo (8%):** {structural_reserve:.2f} €")
    st.write(f"**Reserva real tras operación:** {remaining_cash:.2f} €")
    
    st.divider()
    st.subheader("🧠 Diagnóstico de Mercado")
    if regime == "RISK_ON":
        st.success("🔵 RISK ON: volatilidad baja. Máxima exposición.")
    elif regime == "RISK_OFF":
        st.warning("🟠 RISK OFF: volatilidad alta. Priorizando defensa.")
    elif regime == "ATTACK_MODE":
        st.error("🔴 MODO ATAQUE: BTC en capitulación extrema. Aumentando exposición táctica.")
    else:
        st.info("⚪ NEUTRAL: posicionamiento equilibrado.")
    
    st.caption(f"Última actualización de precios: {prices_df.index[-1].strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    main()
