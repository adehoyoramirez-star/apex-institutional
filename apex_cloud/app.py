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
# CONFIGURACIÓN GENERAL
# ==========================================
st.set_page_config(page_title="Quantum-Plus APEX 150K", layout="wide")

PORTFOLIO_FILE = "portfolio_state.json"
TARGET_GOAL = 150000
STRUCTURAL_RESERVE_PCT = 0.08
DEFAULT_MONTHLY = 400

# Activos con tickers y divisa original (se convertirá a EUR)
ASSETS = {
    "MSCI World Quality": {"ticker": "IWQU.AS", "currency": "EUR"},
    "Emerging Markets IMI": {"ticker": "IS3N.AS", "currency": "EUR"},
    "Global Aggregate Bond": {"ticker": "AGGH.AS", "currency": "EUR"},
    "WisdomTree AI ETF": {"ticker": "WTAI.AS", "currency": "EUR"},
    "Physical Gold": {"ticker": "SGLN.AS", "currency": "EUR"},
    "Uranium ETF": {"ticker": "URNM", "currency": "USD"},      # cotiza en USD
    "Bitcoin": {"ticker": "BTC-USD", "currency": "USD"}        # cotiza en USD
}

# Mapeo sectorial para restricciones (ajústalo según tu criterio)
SECTOR_MAP = {
    "MSCI World Quality": "global_quality",
    "Emerging Markets IMI": "emerging",
    "Global Aggregate Bond": "bonds",
    "WisdomTree AI ETF": "tech",
    "Physical Gold": "gold",
    "Uranium ETF": "uranium",
    "Bitcoin": "crypto"
}

SECTOR_CAP = 0.35  # Límite máximo por sector

# ==========================================
# FUNCIONES DE PERSISTENCIA CON EDITOR
# ==========================================
def load_portfolio():
    """Carga la cartera desde JSON. Si hay error, permite editarlo."""
    # Datos por defecto (ajusta shares y precios medios según tu situación)
    default_positions = {
        "MSCI World Quality": {"shares": 0.0, "avg_price": 0.0},
        "Emerging Markets IMI": {"shares": 0.0, "avg_price": 0.0},
        "Global Aggregate Bond": {"shares": 0.0, "avg_price": 0.0},
        "WisdomTree AI ETF": {"shares": 0.0, "avg_price": 0.0},
        "Physical Gold": {"shares": 0.0, "avg_price": 0.0},
        "Uranium ETF": {"shares": 0.0, "avg_price": 0.0},
        "Bitcoin": {"shares": 0.0, "avg_price": 0.0}
    }
    default = {
        "cash_reserve": 150.0,
        "positions": default_positions,
        "last_updated": datetime.now().isoformat()
    }

    if not os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=4)
        return default

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

    # Asegurar que todas las claves existen
    for name in ASSETS:
        if name not in data["positions"]:
            data["positions"][name] = {"shares": 0.0, "avg_price": 0.0}
    if "cash_reserve" not in data:
        data["cash_reserve"] = 150.0
    return data

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=4)

portfolio = load_portfolio()

# ==========================================
# OBTENER DATOS DE MERCADO (SIN SIMULACIONES)
# ==========================================
@st.cache_data(ttl=300)
def get_market_data():
    """Descarga precios actuales, tipo de cambio y datos macro."""
    # Lista completa de tickers a descargar
    tickers_list = [v["ticker"] for v in ASSETS.values()] + ["EURUSD=X", "^VIX", "^TNX", "^IRX", "^GSPC"]
    try:
        raw = yf.download(tickers_list, period="1d", progress=False)["Close"].iloc[-1]
        # Si algún ticker no se ha descargado, raw será NaN, pero lo dejamos así y manejamos después
    except Exception as e:
        st.error(f"Error al descargar datos de mercado: {e}. Inténtalo más tarde.")
        st.stop()

    # Extraer precios en moneda original
    prices_original = {}
    for name, info in ASSETS.items():
        ticker = info["ticker"]
        if ticker in raw and pd.notna(raw[ticker]):
            prices_original[name] = raw[ticker]
        else:
            st.error(f"No se pudo obtener precio para {name} ({ticker}). Verifica el ticker.")
            st.stop()

    # Tipo de cambio EUR/USD
    if "EURUSD=X" in raw and pd.notna(raw["EURUSD=X"]):
        eurusd = raw["EURUSD=X"]
    else:
        st.error("No se pudo obtener el tipo de cambio EUR/USD.")
        st.stop()

    # Convertir a EUR
    prices_eur = {}
    for name, info in ASSETS.items():
        if info["currency"] == "USD":
            prices_eur[name] = prices_original[name] / eurusd
        else:
            prices_eur[name] = prices_original[name]

    # Datos macro
    macro = {}
    for idx in ["^VIX", "^TNX", "^IRX", "^GSPC"]:
        if idx in raw and pd.notna(raw[idx]):
            macro[idx] = raw[idx]
        else:
            st.error(f"No se pudo obtener el indicador {idx}.")
            st.stop()

    return prices_eur, macro, eurusd

# Cargar datos (si falla, la app se detiene)
prices, macro, eurusd = get_market_data()

# ==========================================
# CALCULAR VALOR DE CARTERA
# ==========================================
def calculate_portfolio():
    total = portfolio["cash_reserve"]
    rows = []
    for name in ASSETS:
        shares = portfolio["positions"][name]["shares"]
        avg_price = portfolio["positions"][name]["avg_price"]
        current_price = prices[name]
        value = shares * current_price
        total += value
        rows.append([name, shares, avg_price, current_price, value])
    df = pd.DataFrame(rows, columns=["Activo", "Shares", "Precio Medio", "Precio Actual", "Valor (EUR)"])
    return total, df

total_value, df_portfolio = calculate_portfolio()

# ==========================================
# RÉGIMEN DE MERCADO (basado en VIX)
# ==========================================
vix = macro["^VIX"]
tnx = macro["^TNX"]
irx = macro["^IRX"]
sp500 = macro["^GSPC"]

# Para calcular percentiles necesitamos serie histórica (la descargamos ahora)
@st.cache_data(ttl=3600)
def get_historical_vix():
    vix_hist = yf.download("^VIX", period="3y", progress=False)["Close"]
    return vix_hist

vix_hist = get_historical_vix()
vix_p80 = vix_hist.quantile(0.8)
vix_p20 = vix_hist.quantile(0.2)

if vix > vix_p80:
    regime = "RISK_OFF"
    target_vol = 0.10
elif vix < vix_p20:
    regime = "RISK_ON"
    target_vol = 0.18
else:
    regime = "NEUTRAL"
    target_vol = 0.14

# ==========================================
# ATAQUE BTC (Z-score)
# ==========================================
@st.cache_data(ttl=3600)
def get_btc_historical():
    btc_hist = yf.download("BTC-USD", period="3y", progress=False)["Close"] / eurusd  # en EUR
    return btc_hist

btc_hist = get_btc_historical()
ma200 = btc_hist.rolling(200).mean()
std200 = btc_hist.rolling(200).std()
btc_z = (btc_hist.iloc[-1] - ma200.iloc[-1]) / std200.iloc[-1]

attack_mode = btc_z < -2
if attack_mode:
    regime = "ATTACK_MODE"
    target_vol = 0.22

# ==========================================
# OPTIMIZACIÓN DE CARTERA (pesos objetivo)
# ==========================================
# Necesitamos datos históricos de retornos para todos los activos (en EUR)
@st.cache_data(ttl=3600)
def get_historical_returns():
    tickers_eur = []
    for name, info in ASSETS.items():
        ticker = info["ticker"]
        if info["currency"] == "USD":
            # Descargar en USD y luego convertir a EUR usando histórico EUR/USD
            # Para simplificar, usamos el tipo de cambio actual como aproximación (no es exacto pero válido para el modelo)
            # Una implementación más precisa requeriría descargar también histórico de EUR/USD.
            # Aceptamos esta simplificación para mantener la agilidad.
            df_usd = yf.download(ticker, period="3y", progress=False)["Close"]
            # Convertimos a EUR usando el tipo actual (podría mejorarse)
            df_eur = df_usd / eurusd
        else:
            df_eur = yf.download(ticker, period="3y", progress=False)["Close"]
        tickers_eur.append(df_eur)
    # Concatenar y calcular retornos
    hist_prices = pd.concat(tickers_eur, axis=1, keys=list(ASSETS.keys()))
    returns = hist_prices.pct_change().dropna()
    return returns

returns = get_historical_returns()
mu = returns.mean() * 252
cov = returns.cov() * 252

# Función de optimización
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
    
    # Límites individuales
    bounds = [(0.02, 0.40) for _ in range(n)]
    btc_idx = names.index("Bitcoin")
    bounds[btc_idx] = (btc_min, btc_max)
    
    w0 = np.ones(n) / n
    result = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints,
                      method='SLSQP', options={'ftol': 1e-6})
    
    if not result.success:
        # Fallback a mínima varianza
        def port_vol(w): return np.sqrt(np.dot(w, np.dot(cov, w)))
        result = minimize(port_vol, w0, bounds=bounds, constraints=constraints, method='SLSQP')
    
    return pd.Series(result.x, index=names)

# ==========================================
# CONTRIBUCIÓN AL RIESGO
# ==========================================
def risk_contribution(weights, cov_matrix):
    port_var = np.dot(weights, np.dot(cov_matrix, weights))
    marginal_contrib = np.dot(cov_matrix, weights)
    risk_contrib = weights * marginal_contrib / np.sqrt(port_var)
    return risk_contrib / risk_contrib.sum()

# ==========================================
# MONTE CARLO (con aportes)
# ==========================================
def run_monte_carlo(current_value, monthly_injection, years, mu_annual, vol_annual, n_sims=1000):
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
# GENERAR ÓRDENES DE COMPRA
# ==========================================
def generate_orders(current_weights, target_weights, current_values, cash_available):
    total_value = sum(current_values.values())
    target_values = {name: target_weights[name] * (total_value + cash_available) for name in target_weights.index}
    orders = {}
    spent = 0
    for name in target_weights.index:
        current = current_values.get(name, 0)
        target = target_values[name]
        diff = target - current
        if diff > 0:
            price = prices[name]
            # Para Bitcoin permitimos fracciones, para ETFs enteros (asumimos que se pueden comprar fracciones? En realidad no, pero lo dejamos como fracción por simplicidad)
            units = diff / price
            if units * price <= cash_available - spent:
                orders[name] = units
                spent += units * price
    return orders, spent

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================
st.title("🚀 **Quantum-Plus APEX 150K ELITE** — Hedge Fund Edition")
st.markdown("---")

# Barra lateral
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

# ==========================================
# MÉTRICAS SUPERIORES
# ==========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("RÉGIMEN", regime, delta=f"VIX {vix:.1f}")
col2.metric("BTC Precio (EUR)", f"{prices['Bitcoin']:,.0f} €", delta=f"Z-score {btc_z:.2f}")

# Calcular pesos actuales
current_values = {name: portfolio["positions"][name]["shares"] * prices[name] for name in ASSETS}
current_total_inv = sum(current_values.values())
if current_total_inv > 0:
    current_weights = pd.Series({name: current_values[name] / current_total_inv for name in ASSETS})
else:
    current_weights = pd.Series({name: 1/len(ASSETS) for name in ASSETS})

# Optimizar pesos objetivo (usando los sliders)
target_weights = optimize_portfolio(btc_min, btc_max)

expected_return = np.dot(target_weights, mu)
mc_base = run_monte_carlo(total_value, monthly_injection, 10, expected_return, target_vol)
prob_base = np.mean(mc_base >= TARGET_GOAL)

col3.metric("Probabilidad 150K", f"{prob_base:.1%}")
col4.metric("Reserva actual", f"{portfolio['cash_reserve']:.2f} €", delta=f"Disponible: ...")

st.markdown("---")

# ==========================================
# TABLA DE POSICIONES
# ==========================================
st.subheader("📊 Cartera Actual")
st.dataframe(df_portfolio.style.format({
    "Shares": "{:.4f}",
    "Precio Medio": "{:.2f}",
    "Precio Actual": "{:.2f}",
    "Valor (EUR)": "{:.2f}"
}), use_container_width=True)

st.markdown("---")

# ==========================================
# GAUGES DE LIQUIDEZ GLOBAL
# ==========================================
st.subheader("🌍 Indicadores de Liquidez Global")
col_g1, col_g2, col_g3, col_g4 = st.columns(4)

# Gauge VIX
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

# Gauge Curva 10y-3m (TED spread aproximado)
ted_spread = tnx - irx
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

# Gauge Tipo 10 años
fig_rate = go.Figure(go.Indicator(
    mode="gauge+number", value=tnx, title="Tipo 10 años (%)",
    number={'suffix': '%'},
    gauge={'axis': {'range': [0, 6]},
           'bar': {'color': "darkgreen"},
           'steps': [{'range': [0, 2], 'color': "lightgreen"},
                     {'range': [2, 4], 'color': "yellow"},
                     {'range': [4, 6], 'color': "red"}]}))
fig_rate.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
col_g3.plotly_chart(fig_rate, use_container_width=True)

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

st.markdown("---")

# ==========================================
# DONUTS: ASIGNACIÓN OBJETIVO Y RIESGO
# ==========================================
col_d1, col_d2 = st.columns(2)

with col_d1:
    st.subheader("🎯 Asignación Objetivo")
    fig_target = px.pie(names=target_weights.index, values=target_weights.values, hole=0.6)
    st.plotly_chart(fig_target, use_container_width=True)

with col_d2:
    st.subheader("📉 Contribución al Riesgo Actual")
    risk_contrib = risk_contribution(current_weights.values, cov)
    risk_df = pd.DataFrame({"Activo": target_weights.index, "Contribución": risk_contrib})
    fig_risk = px.pie(risk_df, names="Activo", values="Contribución", hole=0.6)
    st.plotly_chart(fig_risk, use_container_width=True)

st.markdown("---")

# ==========================================
# TABLA DE DESVIACIÓN VS OBJETIVO
# ==========================================
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

# ==========================================
# MONTE CARLO CON HISTOGRAMA
# ==========================================
st.subheader("📈 Monte Carlo 10 años (escenarios)")

# Escenarios
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

# Histograma
fig_mc = go.Figure()
fig_mc.add_trace(go.Histogram(x=mc_conserv, name="Conservador", opacity=0.5))
fig_mc.add_trace(go.Histogram(x=mc_base, name="Base", opacity=0.5))
fig_mc.add_trace(go.Histogram(x=mc_opt, name="Optimista", opacity=0.5))
fig_mc.add_vline(x=TARGET_GOAL, line_dash="dash", line_color="red", annotation_text="150K")
fig_mc.update_layout(barmode='overlay', title="Distribución de valor final")
st.plotly_chart(fig_mc, use_container_width=True)

st.markdown("---")

# ==========================================
# APORTE MANUAL Y ÓRDENES
# ==========================================
st.subheader("💰 Aporte de Capital")
aporte = st.number_input("Introduce aporte (€)", min_value=0.0, step=50.0, key="aporte_input")
if st.button("Añadir a Reserva"):
    portfolio["cash_reserve"] += aporte
    portfolio["last_updated"] = datetime.now().isoformat()
    save_portfolio(portfolio)
    st.success("Reserva actualizada")
    st.rerun()

st.markdown("---")
st.subheader("🛒 Órdenes sugeridas (rebalanceo)")

# Calcular efectivo disponible teniendo en cuenta reserva estructural
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
        # Ejecutar órdenes
        for name, units in orders.items():
            old = portfolio["positions"][name]
            new_shares = old["shares"] + units
            new_avg = (old["avg_price"] * old["shares"] + units * prices[name]) / new_shares if new_shares > 0 else 0
            portfolio["positions"][name]["shares"] = new_shares
            portfolio["positions"][name]["avg_price"] = new_avg
            portfolio["cash_reserve"] -= units * prices[name]
        portfolio["last_updated"] = datetime.now().isoformat()
        save_portfolio(portfolio)
        st.success("Órdenes ejecutadas. Cartera actualizada.")
        st.rerun()
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
