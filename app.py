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
import requests

# ================== CONFIGURACIÓN INICIAL ==================
st.set_page_config(layout="wide", page_title="APEX 150K ELITE")

# Archivo de persistencia
PORTFOLIO_FILE = "portfolio.json"

# Parámetros fijos
TARGET_GOAL = 150000
STRUCTURAL_RESERVE_PCT = 0.08
BTC_CAP = 0.25
SECTOR_CAP = 0.35  # Límite por temática (ej. semis + uranio juntos no más de 35%)
TICKERS = ["BTC-EUR", "EMXC.DE", "IS3Q.DE", "PPFB.DE", "U3O8.DE", "VVSM.DE", "ZPRR.DE"]

# Mapeo sectorial para control de concentración
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
    """Carga la cartera desde JSON. Si no existe, crea una por defecto."""
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    else:
        # Estructura por defecto (vacía)
        default = {
            "positions": {t: {"shares": 0, "avg_price": 0} for t in TICKERS},
            "cash_reserve": 0,
            "last_updated": datetime.now().isoformat()
        }
        save_portfolio(default)
        return default

def save_portfolio(portfolio):
    """Guarda la cartera en JSON."""
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=2)

# ================== DESCARGA DE DATOS ==================
@st.cache_data(ttl=300)  # Cache 5 minutos
def get_market_data():
    """Descarga precios y datos macro."""
    all_tickers = TICKERS + ["^VIX", "^TNX", "^GSPC", "M2SL"]  # M2SL necesita FRED
    try:
        raw = yf.download(all_tickers, period="5y", auto_adjust=True, progress=False)["Close"]
    except:
        # Fallback a datos más básicos
        raw = yf.download(TICKERS + ["^VIX", "^TNX", "^GSPC"], period="5y", auto_adjust=True, progress=False)["Close"]
        # M2SL lo obtenemos aparte si es necesario (simulamos)
        # En un caso real usarías FRED API, pero para simplificar lo omitimos o usamos un valor fijo
    
    raw = raw.ffill()
    prices = raw[TICKERS]
    macro = raw[["^VIX", "^TNX", "^GSPC"]]
    
    # Calcular ERP aproximado (S&P 500 earnings yield - 10y Treasury)
    # Usamos datos estáticos si no tenemos M2SL (en producción usarías FRED)
    # Simulamos earnings yield ~ 1/PE del S&P 500 (aprox 4.5% ahora)
    earnings_yield = 0.045  # Placeholder
    risk_free = macro["^TNX"].iloc[-1] / 100 if not macro["^TNX"].isna().all() else 0.04
    erp = earnings_yield - risk_free
    
    return prices, macro, erp

# ================== CÁLCULO DE RÉGIMEN ==================
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

# ================== OPTIMIZACIÓN ROBUSTA ==================
def optimize_portfolio(returns, target_vol, btc_cap, sector_map, sector_cap, attack_mode=False):
    """
    Optimización con penalización de concentración sectorial.
    Maximiza Sharpe sujeto a: 
      - Volatilidad <= target_vol
      - Límite individual (ya definido en bounds)
      - Límite por sector (sector_cap)
    """
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    n = len(returns.columns)
    
    # Función objetivo: negativo Sharpe (maximizar Sharpe)
    def neg_sharpe(w):
        port_return = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        return -port_return / port_vol
    
    # Restricciones: suma = 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    # Restricción de volatilidad
    constraints.append({'type': 'ineq', 'fun': lambda w: target_vol - np.sqrt(w @ cov @ w)})
    
    # Restricciones sectoriales (límite superior por sector)
    sectors = list(set(sector_map.values()))
    for sector in sectors:
        indices = [i for i, t in enumerate(returns.columns) if sector_map[t] == sector]
        if indices:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, idx=indices: sector_cap - np.sum(w[idx])
            })
    
    # Límites individuales
    bounds = [(0.02, 0.40) for _ in range(n)]
    btc_idx = returns.columns.get_loc("BTC-EUR")
    # En ataque, permitimos más BTC
    bounds[btc_idx] = (0.02, btc_cap if not attack_mode else 0.40)
    
    # Punto de partida: pesos iguales
    w0 = np.ones(n) / n
    
    # Optimización
    result = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints, 
                      method='SLSQP', options={'ftol': 1e-6})
    
    if not result.success:
        # Fallback a mínima varianza
        def port_vol(w):
            return np.sqrt(w @ cov @ w)
        result = minimize(port_vol, w0, bounds=bounds, constraints=constraints, 
                          method='SLSQP', options={'ftol': 1e-6})
    
    return pd.Series(result.x, index=returns.columns)

# ================== CONTRIBUCIÓN AL RIESGO ==================
def risk_contribution(weights, cov):
    """Calcula la contribución porcentual al riesgo total."""
    port_var = weights @ cov @ weights
    marginal_contrib = cov @ weights
    risk_contrib = weights * marginal_contrib / np.sqrt(port_var)
    return risk_contrib / risk_contrib.sum()  # normalizado

# ================== MONTE CARLO POR ESCENARIOS ==================
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
    """
    Calcula órdenes para acercarse a los pesos objetivo, respetando liquidez.
    """
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

# ================== ACTUALIZAR PORTAFOLIO TRAS COMPRAS ==================
def execute_orders(portfolio, orders, prices):
    """
    Actualiza shares y precio medio, y resta la reserva.
    """
    cash_spent = 0
    for t, units in orders.items():
        price = prices[t]
        old = portfolio["positions"].get(t, {"shares": 0, "avg_price": 0})
        new_shares = old["shares"] + units
        if new_shares > 0:
            new_avg = (old["avg_price"] * old["shares"] + units * price) / new_shares
        else:
            new_avg = 0
        portfolio["positions"][t] = {"shares": new_shares, "avg_price": new_avg}
        cash_spent += units * price
    
    portfolio["cash_reserve"] -= cash_spent
    portfolio["last_updated"] = datetime.now().isoformat()
    save_portfolio(portfolio)
    return portfolio

# ================== INTERFAZ PRINCIPAL ==================
def main():
    st.title("🦅 **APEX 150K ELITE** — HEDGE FUND EDITION")
    
    # Cargar cartera actual
    portfolio = load_portfolio()
    
    # Sidebar para acciones y entradas
    with st.sidebar:
        st.header("⚙️ Controles")
        monthly_injection = st.number_input("Aporte mensual (€)", min_value=0, value=400, step=50)
        
        st.markdown("---")
        st.subheader("💾 Estado cartera")
        st.json({
            "Reserva actual": f"{portfolio['cash_reserve']:.2f} €",
            "Última actualización": portfolio.get("last_updated", "N/A")
        })
        
        if st.button("⟳ Recargar datos de mercado"):
            st.cache_data.clear()
            st.rerun()
    
    # Obtener datos de mercado
    prices_df, macro_df, erp = get_market_data()
    latest_prices = prices_df.iloc[-1]
    
    # Calcular valores actuales
    current_values = {}
    for t in TICKERS:
        pos = portfolio["positions"].get(t, {"shares": 0})
        shares = pos["shares"]
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
        target_vol = 0.22  # Permitir más riesgo
    
    # Optimizar pesos objetivo
    returns = prices_df.pct_change().dropna()
    target_weights = optimize_portfolio(returns, target_vol, BTC_CAP, SECTOR_MAP, SECTOR_CAP, attack_mode)
    
    # Calcular contribución al riesgo actual
    cov = returns.cov() * 252
    risk_contrib = risk_contribution(current_weights.values, cov)
    
    # Disponible para invertir (reserva actual + aporte mensual)
    total_cash = portfolio["cash_reserve"] + monthly_injection
    structural_reserve = STRUCTURAL_RESERVE_PCT * (current_total + monthly_injection)  # reserva objetivo
    usable_cash = max(0, total_cash - structural_reserve)
    
    # En ataque, podemos usar toda la reserva si queremos (pero dejamos la estructural como objetivo)
    if attack_mode:
        usable_cash = total_cash  # Opcional, pero podemos ser más agresivos
    
    # Generar órdenes
    orders, spent = generate_orders(current_weights, target_weights, current_values, usable_cash, latest_prices)
    remaining_cash = total_cash - spent
    
    # ================== DASHBOARD PRINCIPAL ==================
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RÉGIMEN", regime, delta=f"VIX {vix:.1f}")
    with col2:
        st.metric("BTC Precio", f"{latest_prices['BTC-EUR']:,.0f} €", delta=f"Z-score {btc_z:.2f}")
    with col3:
        # Mostrar probabilidad (simulada con escenario base)
        # Usamos retorno esperado del portafolio objetivo
        expected_return = target_weights @ (returns.mean() * 252)
        mc_results_base = run_monte_carlo(current_total, monthly_injection, 10, expected_return, target_vol)
        prob_base = np.mean(mc_results_base >= TARGET_GOAL)
        st.metric("Probabilidad 150K", f"{prob_base:.1%}")
    with col4:
        st.metric("Reserva actual", f"{portfolio['cash_reserve']:.2f} €", delta=f"Disponible: {usable_cash:.2f}")
    
    st.divider()
    
    # ================== GAUGES MACRO (GRANDES) ==================
    st.subheader("📊 Panorama Macro")
    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    
    with col_g1:
        # Gauge VIX
        fig_vix = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = vix,
            title = {'text': "VIX"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 40]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 30], 'color': "yellow"},
                    {'range': [30, 40], 'color': "red"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': vix_p80 if 'vix_p80' in locals() else 25}}))
        fig_vix.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_vix, use_container_width=True)
    
    with col_g2:
        # ERP aproximado (usamos un valor fijo por simplicidad, pero podrías calcularlo con datos reales)
        # En un hedge fund real usarías FRED para earnings yield
        earnings_yield = 0.045  # Placeholder
        risk_free = macro_df["^TNX"].iloc[-1] / 100 if not pd.isna(macro_df["^TNX"].iloc[-1]) else 0.04
        erp_value = earnings_yield - risk_free
        fig_erp = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = erp_value * 100,  # en %
            title = {'text': "ERP (earnings yield - 10y)"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [-2, 6]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [-2, 1], 'color': "red"},
                    {'range': [1, 3], 'color': "yellow"},
                    {'range': [3, 6], 'color': "lightgreen"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 2.5}}))
        fig_erp.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_erp, use_container_width=True)
    
    with col_g3:
        # Drawdown S&P 500
        sp500 = macro_df["^GSPC"]
        rolling_max = sp500.expanding().max()
        drawdown = (sp500 - rolling_max) / rolling_max * 100
        current_dd = drawdown.iloc[-1]
        fig_dd = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = current_dd,
            title = {'text': "S&P 500 Drawdown %"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'suffix': "%"},
            gauge = {
                'axis': {'range': [-50, 0]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [-10, 0], 'color': "lightgreen"},
                    {'range': [-20, -10], 'color': "yellow"},
                    {'range': [-50, -20], 'color': "red"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': -15}}))
        fig_dd.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_dd, use_container_width=True)
    
    with col_g4:
        # Z-score BTC
        fig_z = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = btc_z,
            title = {'text': "BTC Z-score (200d)"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [-3, 3]},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [-1, 1], 'color': "lightgreen"},
                    {'range': [1, 2], 'color': "yellow"},
                    {'range': [2, 3], 'color': "red"},
                    {'range': [-2, -1], 'color': "yellow"},
                    {'range': [-3, -2], 'color': "red"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': -2}}))
        fig_z.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_z, use_container_width=True)
    
    st.divider()
    
    # ================== DONUT OBJETIVO VS ACTUAL Y CONTRIBUCIÓN AL RIESGO ==================
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.subheader("🎯 Asignación Objetivo")
        fig_target = px.pie(
            names=target_weights.index,
            values=target_weights.values,
            hole=0.6,
            title="Pesos objetivo"
        )
        st.plotly_chart(fig_target, use_container_width=True)
    
    with col_d2:
        st.subheader("📉 Contribución al Riesgo Actual")
        risk_df = pd.DataFrame({
            "Activo": target_weights.index,
            "Contribución": risk_contrib
        })
        fig_risk = px.pie(
            risk_df,
            names="Activo",
            values="Contribución",
            hole=0.6,
            title="Risk contribution (actual)"
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Tabla de desviaciones
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
    # Escenario conservador: retorno esperado -2%, vol +2%
    mu_conserv = expected_return - 0.02
    vol_conserv = target_vol + 0.02
    # Escenario base
    mu_base = expected_return
    vol_base = target_vol
    # Escenario optimista
    mu_opt = expected_return + 0.02
    vol_opt = target_vol - 0.02
    
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
    
    # Histogramas
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Histogram(x=mc_conserv, name="Conservador", opacity=0.5))
    fig_mc.add_trace(go.Histogram(x=mc_base, name="Base", opacity=0.5))
    fig_mc.add_trace(go.Histogram(x=mc_opt, name="Optimista", opacity=0.5))
    fig_mc.add_vline(x=TARGET_GOAL, line_dash="dash", line_color="red", annotation_text="150K")
    fig_mc.update_layout(barmode='overlay', title="Distribución de valor final")
    st.plotly_chart(fig_mc, use_container_width=True)
    
    st.divider()
    
    # ================== ÓRDENES Y EJECUCIÓN ==================
    st.subheader("🛒 Órdenes sugeridas")
    if orders:
        for t, units in orders.items():
            cost = units * latest_prices[t]
            st.write(f"• **{t}**: comprar {units} unidades a {latest_prices[t]:.2f} € → coste {cost:.2f} €")
        st.write(f"**Coste total:** {spent:.2f} €")
        st.write(f"**Reserva restante tras compras:** {remaining_cash:.2f} €")
        
        if st.button("✅ Confirmar ejecución"):
            # Ejecutar órdenes
            portfolio = execute_orders(portfolio, orders, latest_prices)
            # Actualizar reserva (ya se resta dentro de execute_orders)
            st.success("Órdenes ejecutadas. Cartera actualizada.")
            st.rerun()
    else:
        st.info("No hay órdenes generadas (saldo insuficiente o cartera ya equilibrada).")
    
    # Mostrar reserva estructural
    st.write(f"**Reserva estructural recomendada (8%):** {structural_reserve:.2f} €")
    st.write(f"**Reserva actual tras operación:** {remaining_cash:.2f} €")
    
    st.divider()
    
    # ================== DIAGNÓSTICO ==================
    st.subheader("🧠 Diagnóstico de Mercado")
    if regime == "RISK_ON":
        st.success("🔵 Régimen RISK ON: volatilidad baja, momento favorable. Máxima exposición.")
    elif regime == "RISK_OFF":
        st.warning("🟠 Régimen RISK OFF: volatilidad alta. Priorizando defensa y reducción de riesgo.")
    elif regime == "ATTACK_MODE":
        st.error("🔴 MODO ATAQUE: BTC en capitulación extrema. Aumentando exposición táctica.")
    else:
        st.info("⚪ Régimen NEUTRAL: posicionamiento equilibrado.")
    
    st.caption(f"Última actualización de precios: {prices_df.index[-1].strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    main()
