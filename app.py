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
# CONFIGURACIÓN
# ==========================================
st.set_page_config(page_title="APEX 150K ELITE - Ultra Rápido", layout="wide")

PORTFOLIO_FILE = "portfolio.json"
TARGET_GOAL = 150000
STRUCTURAL_RESERVE_PCT = 0.08
DEFAULT_MONTHLY = 400

# TUS TICKERS REALES
TICKERS = ["BTC-EUR", "EMXC.DE", "IS3Q.DE", "PPFB.DE", "URNU.DE", "VVSM.DE", "ZPRR.DE"]

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
# FUNCIONES DE PERSISTENCIA
# ==========================================
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    else:
        default = {
            "positions": {t: {"shares": 0.0, "avg_price": 0.0} for t in TICKERS},
            "cash_reserve": 150.0,
            "last_updated": datetime.now().isoformat()
        }
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(default, f, indent=4)
        return default

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=4)

portfolio = load_portfolio()

# ==========================================
# DATOS CACHEADOS (descarga única pesada)
# ==========================================
@st.cache_data(ttl=3600)  # 1 hora de caché
def get_historical_data():
    """Descarga históricos de 2 años para todos los activos + VIX. Solo una vez."""
    all_tickers = TICKERS + ["^VIX", "^TNX", "^IRX"]
    try:
        data = yf.download(all_tickers, period="2y", auto_adjust=True, progress=False)["Close"]
        data = data.ffill().bfill()  # rellenar NaNs
        return data
    except Exception as e:
        st.error(f"Error al descargar datos históricos: {e}")
        return None

@st.cache_data(ttl=300)  # 5 minutos para precios actuales
def get_current_prices():
    """Solo precios de hoy (rápido)."""
    all_tickers = TICKERS + ["^VIX", "^TNX", "^IRX", "^GSPC"]
    try:
        data = yf.download(all_tickers, period="1d", progress=False)["Close"].iloc[-1]
        return data
    except Exception as e:
        st.error(f"Error al obtener precios actuales: {e}")
        return None

# Cargar datos históricos (en segundo plano, cacheado)
hist_data = get_historical_data()
if hist_data is None:
    st.stop()

# Cargar precios actuales
current = get_current_prices()
if current is None:
    st.stop()

# Extraer precios de activos (todos en EUR)
prices = {ticker: current[ticker] for ticker in TICKERS}

# Datos macro actuales
vix = current["^VIX"]
tnx = current["^TNX"]
irx = current["^IRX"]
gspc = current["^GSPC"]

# Calcular percentiles del VIX a partir del histórico (ya cacheado)
vix_hist = hist_data["^VIX"]
vix_p80 = vix_hist.quantile(0.8)
vix_p20 = vix_hist.quantile(0.2)

# Régimen de mercado
if vix > vix_p80:
    regime = "RISK_OFF"
    target_vol = 0.10
elif vix < vix_p20:
    regime = "RISK_ON"
    target_vol = 0.18
else:
    regime = "NEUTRAL"
    target_vol = 0.14

# Z-score de BTC (necesitamos serie histórica de BTC)
btc_hist = hist_data["BTC-EUR"]
ma200 = btc_hist.rolling(200).mean()
std200 = btc_hist.rolling(200).std()
btc_z = (btc_hist.iloc[-1] - ma200.iloc[-1]) / std200.iloc[-1]

attack_mode = btc_z < -2
if attack_mode:
    regime = "ATTACK_MODE"
    target_vol = 0.22

# ==========================================
# VALORES ACTUALES DE CARTERA
# ==========================================
current_values = {}
for ticker in TICKERS:
    shares = portfolio["positions"].get(ticker, {}).get("shares", 0)
    current_values[ticker] = shares * prices[ticker]

total_invested = sum(current_values.values())
total_value = total_invested + portfolio["cash_reserve"]

if total_invested > 0:
    current_weights = pd.Series({t: current_values[t]/total_invested for t in TICKERS})
else:
    current_weights = pd.Series({t: 1/len(TICKERS) for t in TICKERS})

# ==========================================
# INTERFAZ PRINCIPAL (carga instantánea)
# ==========================================
st.title("🚀 **APEX 150K ELITE — Optimizado**")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Controles")
    monthly_injection = st.number_input("Aporte mensual (€)", min_value=0, value=DEFAULT_MONTHLY, step=50)
    btc_min = st.slider("Peso mínimo BTC", 0.0, 0.4, 0.2, 0.01)
    btc_max = st.slider("Peso máximo BTC", btc_min, 0.4, 0.3, 0.01)
    st.markdown("---")
    st.subheader("💾 Estado")
    st.json({"Reserva": f"{portfolio['cash_reserve']:.2f} €"})
    if st.button("⟳ Recargar precios"):
        st.cache_data.clear()
        st.rerun()

# Métricas superiores
col1, col2, col3, col4 = st.columns(4)
col1.metric("RÉGIMEN", regime, delta=f"VIX {vix:.1f}")
col2.metric("BTC Precio", f"{prices['BTC-EUR']:,.0f} €", delta=f"Z {btc_z:.2f}")
col3.metric("Valor Cartera", f"{total_value:,.0f} €")
col4.metric("Reserva", f"{portfolio['cash_reserve']:.2f} €")

st.markdown("---")

# Tabla de posiciones
df = pd.DataFrame([
    [ticker,
     portfolio["positions"][ticker]["shares"],
     prices[ticker],
     current_values[ticker]]
    for ticker in TICKERS
], columns=["Activo", "Shares", "Precio", "Valor (EUR)"])
st.dataframe(df.style.format({"Shares": "{:.4f}", "Precio": "{:.2f}", "Valor (EUR)": "{:.2f}"}))

st.markdown("---")

# ==========================================
# ANÁLISIS AVANZADO (usa datos cacheados)
# ==========================================
if st.button("📊 Calcular optimización y Monte Carlo (rápido)"):
    with st.spinner("Calculando... (usa datos cacheados, será instantáneo)"):
        # Construir matriz de retornos desde historial cacheado
        hist_assets = hist_data[TICKERS]
        returns = hist_assets.pct_change().dropna()
        mu = returns.mean() * 252
        cov = returns.cov() * 252

        # Optimización
        n = len(TICKERS)
        def neg_sharpe(w):
            return - (w @ mu) / np.sqrt(w @ cov @ w)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w)-1}]
        constraints.append({'type': 'ineq', 'fun': lambda w: target_vol - np.sqrt(w @ cov @ w)})

        # Restricciones sectoriales
        for sector in set(SECTOR_MAP.values()):
            idx = [i for i, t in enumerate(TICKERS) if SECTOR_MAP[t] == sector]
            if idx:
                constraints.append({'type': 'ineq', 'fun': lambda w, i=idx: SECTOR_CAP - np.sum(w[i])})

        bounds = [(0.02, 0.4)] * n
        btc_idx = TICKERS.index("BTC-EUR")
        bounds[btc_idx] = (btc_min, btc_max)

        w0 = np.ones(n)/n
        result = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints, method='SLSQP')
        target_weights = pd.Series(result.x, index=TICKERS)
        expected_return = target_weights @ mu

        # Monte Carlo vectorizado
        months = 10 * 12
        monthly_mu = expected_return / 12
        monthly_vol = target_vol / np.sqrt(12)
        n_sims = 1000

        # Generar todas las rentabilidades mensuales de una vez
        random_returns = np.random.normal(monthly_mu, monthly_vol, (n_sims, months))
        # Calcular factor de crecimiento acumulado
        growth = np.cumprod(1 + random_returns, axis=1)
        # Aportaciones mensuales (se invierten al final de cada mes)
        contributions = monthly_injection * np.sum(np.cumprod(1 + random_returns[:, ::-1], axis=1)[:, ::-1], axis=1)
        # Valor final = valor inicial * crecimiento + suma de aportes capitalizados
        final_values = total_value * growth[:, -1] + contributions

        prob = np.mean(final_values >= TARGET_GOAL)

        # Mostrar resultados
        st.success(f"**Probabilidad de alcanzar 150k: {prob:.1%}**")
        st.metric("Rentabilidad esperada anual", f"{expected_return:.1%}")
        st.metric("Volatilidad objetivo", f"{target_vol:.1%}")

        # Gráficos
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🎯 Asignación objetivo")
            fig = px.pie(names=target_weights.index, values=target_weights.values, hole=0.6)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.subheader("📉 Contribución al riesgo actual")
            risk_contrib = (current_weights * (cov @ current_weights)) / np.sqrt(current_weights @ cov @ current_weights)
            risk_contrib = risk_contrib / risk_contrib.sum()
            fig2 = px.pie(names=TICKERS, values=risk_contrib, hole=0.6)
            st.plotly_chart(fig2, use_container_width=True)

        # Histograma Monte Carlo
        st.subheader("📈 Distribución del valor final en 10 años")
        fig3 = go.Figure(data=[go.Histogram(x=final_values, nbinsx=50)])
        fig3.add_vline(x=TARGET_GOAL, line_dash="dash", line_color="red", annotation_text="150K")
        st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# APORTE MANUAL Y ÓRDENES (rápido)
# ==========================================
st.markdown("---")
st.subheader("💰 Aporte manual")
aporte = st.number_input("Cantidad (€)", 0.0, step=50.0, key="aporte")
if st.button("Añadir a reserva"):
    portfolio["cash_reserve"] += aporte
    portfolio["last_updated"] = datetime.now().isoformat()
    save_portfolio(portfolio)
    st.success("Reserva actualizada. Recarga para ver cambios.")
    st.rerun()  # solo aquí usamos rerun porque es rápido

# Órdenes sugeridas (sin descargar nada, solo usando precios actuales)
st.markdown("---")
st.subheader("🛒 Órdenes sugeridas")
total_cash = portfolio["cash_reserve"] + monthly_injection
structural_reserve = STRUCTURAL_RESERVE_PCT * (total_value + monthly_injection)
usable_cash = total_cash if attack_mode else max(0, total_cash - structural_reserve)

# Para generar órdenes necesitamos pesos objetivo (los calculamos con los parámetros actuales)
# Pero para evitar duplicar código, podemos llamar a la optimización aquí también (rápida porque ya tenemos returns)
# O podríamos guardar target_weights en session_state, pero por simplicidad lo recalculamos.
if 'target_weights' not in st.session_state:
    # recálculo rápido con datos cacheados
    hist_assets = hist_data[TICKERS]
    returns = hist_assets.pct_change().dropna()
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    n = len(TICKERS)
    def neg_sharpe(w):
        return - (w @ mu) / np.sqrt(w @ cov @ w)
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w)-1},
                   {'type': 'ineq', 'fun': lambda w: target_vol - np.sqrt(w @ cov @ w)}]
    for sector in set(SECTOR_MAP.values()):
        idx = [i for i, t in enumerate(TICKERS) if SECTOR_MAP[t] == sector]
        if idx:
            constraints.append({'type': 'ineq', 'fun': lambda w, i=idx: SECTOR_CAP - np.sum(w[i])})
    bounds = [(0.02, 0.4)] * n
    bounds[TICKERS.index("BTC-EUR")] = (btc_min, btc_max)
    w0 = np.ones(n)/n
    result = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints, method='SLSQP')
    st.session_state.target_weights = pd.Series(result.x, index=TICKERS)

target_weights = st.session_state.target_weights

# Generar órdenes
def generate_orders():
    total_inv = total_invested
    target_values = {t: target_weights[t] * (total_inv + usable_cash) for t in TICKERS}
    orders = {}
    spent = 0
    for t in TICKERS:
        current = current_values[t]
        target = target_values[t]
        diff = target - current
        if diff > 0:
            units = diff / prices[t]
            if units * prices[t] <= usable_cash - spent:
                orders[t] = units
                spent += units * prices[t]
    return orders, spent

orders, spent = generate_orders()

if orders:
    for t, units in orders.items():
        st.write(f"• **{t}**: comprar {units:.4f} unidades a {prices[t]:.2f} € → coste {units*prices[t]:.2f} €")
    st.write(f"**Coste total:** {spent:.2f} €")
    st.write(f"**Reserva restante tras compras:** {total_cash - spent:.2f} €")
    if st.button("✅ Confirmar ejecución"):
        for t, units in orders.items():
            old = portfolio["positions"][t]
            new_shares = old["shares"] + units
            new_avg = (old["avg_price"] * old["shares"] + units * prices[t]) / new_shares
            portfolio["positions"][t] = {"shares": new_shares, "avg_price": new_avg}
        portfolio["cash_reserve"] -= spent
        portfolio["last_updated"] = datetime.now().isoformat()
        save_portfolio(portfolio)
        st.success("Órdenes ejecutadas. Recarga para ver cambios.")
        st.rerun()
else:
    st.info("No hay órdenes generadas (saldo insuficiente o cartera equilibrada).")

st.write(f"**Reserva estructural (8%):** {structural_reserve:.2f} €")
st.caption(f"Última actualización de precios: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
