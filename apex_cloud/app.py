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

# ==========================================
# CONFIGURACIÓN - TUS TICKERS REALES
# ==========================================
st.set_page_config(page_title="APEX 150K ELITE - Real", layout="wide")

PORTFOLIO_FILE = "portfolio.json"
TARGET_GOAL = 150000
STRUCTURAL_RESERVE_PCT = 0.08
DEFAULT_MONTHLY = 400

# Activos (coinciden exactamente con tu JSON)
ASSETS = {
    "BTC-EUR": "BTC-EUR",
    "EMXC.DE": "EMXC.DE",
    "IS3Q.DE": "IS3Q.DE",
    "PPFB.DE": "PPFB.DE",
    "URNU.DE": "URNU.DE",
    "VVSM.DE": "VVSM.DE",
    "ZPRR.DE": "ZPRR.DE"
}

# Mapeo sectorial (ajústalo si quieres)
SECTOR_MAP = {
    "BTC-EUR": "crypto",
    "EMXC.DE": "emerging",
    "IS3Q.DE": "emerging",
    "PPFB.DE": "gold",
    "URNU.DE": "uranium",
    "VVSM.DE": "semis",
    "ZPRR.DE": "real_estate"
}
SECTOR_CAP = 0.35

# ==========================================
# FUNCIONES DE PERSISTENCIA (con editor JSON)
# ==========================================
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            st.error(f"**Error en el archivo `{PORTFOLIO_FILE}`:** {str(e)}")
            st.markdown("### Edita el contenido y pulsa Guardar (luego recarga manualmente)")
            with open(PORTFOLIO_FILE, "r", encoding="utf-8", errors="ignore") as f:
                raw_content = f.read()
            new_content = st.text_area("Contenido actual (corrígelo si es necesario):", raw_content, height=400)
            if st.button("💾 Guardar (sin recargar automáticamente)"):
                try:
                    json.loads(new_content)
                    backup_name = f"portfolio_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(backup_name, "w", encoding="utf-8") as f:
                        f.write(raw_content)
                    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    st.success("Archivo guardado correctamente. Recarga la página con el botón ⟳ de la barra lateral.")
                    st.stop()
                except json.JSONDecodeError as e2:
                    st.error(f"El JSON sigue siendo inválido: {e2}. Corrígelo y vuelve a intentar.")
            st.stop()
        return data
    else:
        # Crear archivo por defecto con tus posiciones
        default = {
            "positions": {t: {"shares": 0.0, "avg_price": 0.0} for t in ASSETS},
            "cash_reserve": 150.0,
            "last_updated": datetime.now().isoformat()
        }
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=4)
        return default

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=4)

portfolio = load_portfolio()

# ==========================================
# CARGA DE DATOS TODO-EN-UNO (RÁPIDA)
# ==========================================
@st.cache_data(ttl=3600)
def load_all_data():
    tickers = list(ASSETS.values()) + ["^VIX", "^TNX", "^IRX", "^GSPC"]
    try:
        hist = yf.download(tickers, period="2y", auto_adjust=True, progress=False)["Close"].ffill().bfill()
        latest = hist.iloc[-1]
    except Exception as e:
        st.error(f"Error al descargar datos de Yahoo Finance: {e}")
        st.stop()

    # Precios actuales (todos en EUR, ya que los tickers son EUR)
    prices = {name: latest[ticker] for name, ticker in ASSETS.items()}

    # Datos macro
    macro = {k: latest[k] for k in ["^VIX", "^TNX", "^IRX", "^GSPC"]}

    # DataFrame histórico de activos para retornos
    hist_assets = hist[list(ASSETS.values())]
    hist_assets.columns = list(ASSETS.keys())
    returns = hist_assets.pct_change().dropna()

    # Z-score de Bitcoin
    btc_series = hist_assets["BTC-EUR"]
    ma200 = btc_series.rolling(200).mean()
    std200 = btc_series.rolling(200).std()
    btc_z = (btc_series.iloc[-1] - ma200.iloc[-1]) / std200.iloc[-1]

    # Percentiles VIX
    vix_hist = hist["^VIX"]
    vix_p80 = vix_hist.quantile(0.8)
    vix_p20 = vix_hist.quantile(0.2)

    return {
        "prices": prices,
        "macro": macro,
        "returns": returns,
        "btc_z": btc_z,
        "vix_p80": vix_p80,
        "vix_p20": vix_p20,
        "hist_assets": hist_assets
    }

data = load_all_data()
prices = data["prices"]
macro = data["macro"]
returns = data["returns"]
btc_z = data["btc_z"]
vix_p80 = data["vix_p80"]
vix_p20 = data["vix_p20"]

# ==========================================
# CÁLCULOS INICIALES
# ==========================================
current_values = {}
for name in ASSETS:
    shares = portfolio["positions"].get(name, {}).get("shares", 0)
    current_values[name] = shares * prices[name]

total_invested = sum(current_values.values())
total_value = total_invested + portfolio["cash_reserve"]

if total_invested > 0:
    current_weights = pd.Series({name: current_values[name]/total_invested for name in ASSETS})
else:
    current_weights = pd.Series({name: 1/len(ASSETS) for name in ASSETS})

# Régimen de mercado
vix = macro["^VIX"]
if vix > vix_p80:
    regime = "RISK_OFF"
    target_vol = 0.10
elif vix < vix_p20:
    regime = "RISK_ON"
    target_vol = 0.18
else:
    regime = "NEUTRAL"
    target_vol = 0.14

attack_mode = btc_z < -2
if attack_mode:
    regime = "ATTACK_MODE"
    target_vol = 0.22

# ==========================================
# OPTIMIZACIÓN DE CARTERA
# ==========================================
mu = returns.mean() * 252
cov = returns.cov() * 252

def optimize_portfolio(btc_min, btc_max):
    n = len(ASSETS)
    names = list(ASSETS.keys())

    def neg_sharpe(w):
        port_return = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        return -port_return / port_vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    constraints.append({'type': 'ineq', 'fun': lambda w: target_vol - np.sqrt(np.dot(w, np.dot(cov, w)))})

    # Restricciones sectoriales
    for sector in set(SECTOR_MAP.values()):
        indices = [i for i, name in enumerate(names) if SECTOR_MAP[name] == sector]
        if indices:
            constraints.append({'type': 'ineq', 'fun': lambda w, idx=indices: SECTOR_CAP - np.sum(w[idx])})

    bounds = [(0.02, 0.40) for _ in range(n)]
    btc_idx = names.index("BTC-EUR")
    bounds[btc_idx] = (btc_min, btc_max)

    w0 = np.ones(n) / n
    result = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints,
                      method='SLSQP', options={'ftol': 1e-6})

    if not result.success:
        def port_vol(w): return np.sqrt(np.dot(w, np.dot(cov, w)))
        result = minimize(port_vol, w0, bounds=bounds, constraints=constraints, method='SLSQP')

    return pd.Series(result.x, index=names)

# ==========================================
# MONTE CARLO (500 simulaciones)
# ==========================================
def run_monte_carlo(current_value, monthly_injection, years, mu_annual, vol_annual, n_sims=500):
    months = years * 12
    monthly_mu = mu_annual / 12
    monthly_vol = vol_annual / np.sqrt(12)
    results = []
    for _ in range(n_sims):
        value = current_value
        for m in range(months):
            ret = np.random.normal(monthly_mu, monthly_vol)
            value = value * (1 + ret) + monthly_injection
        results.append(value)
    return np.array(results)

# ==========================================
# GENERAR ÓRDENES
# ==========================================
def generate_orders(current_weights, target_weights, current_values, cash_available):
    total_inv = sum(current_values.values())
    target_values = {name: target_weights[name] * (total_inv + cash_available) for name in target_weights.index}
    orders = {}
    spent = 0
    for name in target_weights.index:
        current = current_values.get(name, 0)
        target = target_values[name]
        diff = target - current
        if diff > 0:
            price = prices[name]
            units = diff / price
            if units * price <= cash_available - spent:
                orders[name] = units
                spent += units * price
    return orders, spent

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================
st.title("🚀 **APEX 150K ELITE** — Tu Cartera Real")
st.markdown("---")

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

# Pesos objetivo
target_weights = optimize_portfolio(btc_min, btc_max)
expected_return = target_weights @ mu
mc_base = run_monte_carlo(total_value, monthly_injection, 10, expected_return, target_vol)
prob_base = np.mean(mc_base >= TARGET_GOAL)

col1, col2, col3, col4 = st.columns(4)
col1.metric("RÉGIMEN", regime, delta=f"VIX {vix:.1f}")
col2.metric("BTC Precio (EUR)", f"{prices['BTC-EUR']:,.0f} €", delta=f"Z-score {btc_z:.2f}")
col3.metric("Probabilidad 150K", f"{prob_base:.1%}")
col4.metric("Reserva actual", f"{portfolio['cash_reserve']:.2f} €")

st.markdown("---")

# Tabla de cartera
df_portfolio = pd.DataFrame([
    [name,
     portfolio["positions"].get(name, {}).get("shares", 0),
     portfolio["positions"].get(name, {}).get("avg_price", 0),
     prices[name],
     current_values[name]]
    for name in ASSETS
], columns=["Activo", "Shares", "Precio Medio", "Precio Actual", "Valor (EUR)"])
st.subheader("📊 Cartera Actual")
st.dataframe(df_portfolio.style.format({
    "Shares": "{:.4f}",
    "Precio Medio": "{:.2f}",
    "Precio Actual": "{:.2f}",
    "Valor (EUR)": "{:.2f}"
}), use_container_width=True)

st.markdown("---")

# Gauges de liquidez global
st.subheader("🌍 Indicadores de Liquidez Global")
col_g1, col_g2, col_g3, col_g4 = st.columns(4)

fig_vix = go.Figure(go.Indicator(
    mode="gauge+number", value=vix, title="VIX",
    gauge={'axis': {'range': [0, 40]},
           'bar': {'color': "darkblue"},
           'steps': [{'range': [0, 20], 'color': "lightgreen"},
                     {'range': [20, 30], 'color': "yellow"},
                     {'range': [30, 40], 'color': "red"}],
           'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': vix_p80}}))
fig_vix.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
col_g1.plotly_chart(fig_vix, use_container_width=True)

ted_spread = macro["^TNX"] - macro["^IRX"]
fig_ted = go.Figure(go.Indicator(
    mode="gauge+number", value=ted_spread*100, title="Curva 10y-3m (pb)",
    number={'suffix': ' pb'},
    gauge={'axis': {'range': [-100, 200]},
           'bar': {'color': "darkred"},
           'steps': [{'range': [-100, 0], 'color': "red"},
                     {'range': [0, 100], 'color': "yellow"},
                     {'range': [100, 200], 'color': "lightgreen"}],
           'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}}))
fig_ted.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
col_g2.plotly_chart(fig_ted, use_container_width=True)

fig_rate = go.Figure(go.Indicator(
    mode="gauge+number", value=macro["^TNX"], title="Tipo 10 años (%)",
    number={'suffix': '%'},
    gauge={'axis': {'range': [0, 6]},
           'bar': {'color': "darkgreen"},
           'steps': [{'range': [0, 2], 'color': "lightgreen"},
                     {'range': [2, 4], 'color': "yellow"},
                     {'range': [4, 6], 'color': "red"}]}))
fig_rate.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
col_g3.plotly_chart(fig_rate, use_container_width=True)

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

st.markdown("---")

# Donuts
col_d1, col_d2 = st.columns(2)
with col_d1:
    st.subheader("🎯 Asignación Objetivo")
    fig_target = px.pie(names=target_weights.index, values=target_weights.values, hole=0.6)
    st.plotly_chart(fig_target, use_container_width=True)

with col_d2:
    st.subheader("📉 Contribución al Riesgo Actual")
    risk_contrib = (current_weights * (cov @ current_weights)) / np.sqrt(current_weights @ cov @ current_weights)
    risk_contrib = risk_contrib / risk_contrib.sum()
    risk_df = pd.DataFrame({"Activo": target_weights.index, "Contribución": risk_contrib})
    fig_risk = px.pie(risk_df, names="Activo", values="Contribución", hole=0.6)
    st.plotly_chart(fig_risk, use_container_width=True)

st.markdown("---")

# Tabla de desviación
st.subheader("📋 Desviación vs Objetivo")
df_compare = pd.DataFrame({
    "Objetivo": target_weights,
    "Actual": current_weights,
    "Diferencia": current_weights - target_weights,
    "Valor actual (€)": [current_values[name] for name in target_weights.index],
    "Precio (EUR)": [prices[name] for name in target_weights.index]
})
st.dataframe(df_compare.style.format({
    "Objetivo": "{:.2%}",
    "Actual": "{:.2%}",
    "Diferencia": "{:.2%}",
    "Valor actual (€)": "{:.2f}",
    "Precio (EUR)": "{:.2f}"
}), use_container_width=True)

st.markdown("---")

# Monte Carlo
st.subheader("📈 Monte Carlo 10 años (escenarios)")
mu_conserv = expected_return - 0.02
vol_conserv = target_vol + 0.02
mu_opt = expected_return + 0.02
vol_opt = target_vol - 0.02

mc_conserv = run_monte_carlo(total_value, monthly_injection, 10, mu_conserv, vol_conserv)
mc_base = run_monte_carlo(total_value, monthly_injection, 10, expected_return, target_vol)
mc_opt = run_monte_carlo(total_value, monthly_injection, 10, mu_opt, vol_opt)

prob_conserv = np.mean(mc_conserv >= TARGET_GOAL)
prob_base = np.mean(mc_base >= TARGET_GOAL)
prob_opt = np.mean(mc_opt >= TARGET_GOAL)

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Conservador", f"{prob_conserv:.1%}")
col_m2.metric("Base", f"{prob_base:.1%}")
col_m3.metric("Optimista", f"{prob_opt:.1%}")

fig_mc = go.Figure()
fig_mc.add_trace(go.Histogram(x=mc_conserv, name="Conservador", opacity=0.5))
fig_mc.add_trace(go.Histogram(x=mc_base, name="Base", opacity=0.5))
fig_mc.add_trace(go.Histogram(x=mc_opt, name="Optimista", opacity=0.5))
fig_mc.add_vline(x=TARGET_GOAL, line_dash="dash", line_color="red", annotation_text="150K")
fig_mc.update_layout(barmode='overlay', title="Distribución de valor final")
st.plotly_chart(fig_mc, use_container_width=True)

st.markdown("---")

# Aporte manual
st.subheader("💰 Aporte de Capital")
aporte = st.number_input("Introduce aporte (€)", min_value=0.0, step=50.0, key="aporte_input")
if st.button("Añadir a Reserva"):
    portfolio["cash_reserve"] += aporte
    portfolio["last_updated"] = datetime.now().isoformat()
    save_portfolio(portfolio)
    st.success("Reserva actualizada. Recarga la página para ver cambios.")
    # No hacemos rerun automático

st.markdown("---")

# Órdenes sugeridas
st.subheader("🛒 Órdenes sugeridas (rebalanceo)")
total_cash = portfolio["cash_reserve"] + monthly_injection
structural_reserve = STRUCTURAL_RESERVE_PCT * (total_value + monthly_injection)
usable_cash = total_cash if attack_mode else max(0, total_cash - structural_reserve)

orders, spent = generate_orders(current_weights, target_weights, current_values, usable_cash)

if orders:
    for name, units in orders.items():
        cost = units * prices[name]
        st.write(f"• **{name}**: comprar {units:.4f} unidades a {prices[name]:.2f} € → coste {cost:.2f} €")
    st.write(f"**Coste total:** {spent:.2f} €")
    st.write(f"**Reserva restante tras compras:** {total_cash - spent:.2f} €")

    if st.button("✅ Confirmar ejecución"):
        for name, units in orders.items():
            old = portfolio["positions"].get(name, {"shares": 0, "avg_price": 0})
            new_shares = old["shares"] + units
            new_avg = (old["avg_price"] * old["shares"] + units * prices[name]) / new_shares if new_shares > 0 else 0
            portfolio["positions"][name] = {"shares": new_shares, "avg_price": new_avg}
            portfolio["cash_reserve"] -= units * prices[name]
        portfolio["last_updated"] = datetime.now().isoformat()
        save_portfolio(portfolio)
        st.success("Órdenes ejecutadas. Recarga la página para ver cambios.")
else:
    st.info("No hay órdenes generadas (saldo insuficiente o cartera ya equilibrada).")

st.write(f"**Reserva estructural objetivo (8%):** {structural_reserve:.2f} €")
st.write(f"**Reserva real tras operación:** {total_cash - spent:.2f} €")

st.markdown("---")
st.subheader("🧠 Diagnóstico de Mercado")
if regime == "RISK_ON":
    st.success("🔵 RISK ON: volatilidad baja. Máxima exposición.")
elif regime == "RISK_OFF":
    st.warning("🟠 RISK OFF: volatilidad alta. Priorizando defensa.")
elif regime == "ATTACK_MODE":
    st.error("🔴 MODO ATAQUE: BTC en capitulación extrema. Aumentando exposición táctica.")
else:
    st.info("⚪ NEUTRAL: posicionamiento equilibrado.")

st.caption(f"Última actualización de precios: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
