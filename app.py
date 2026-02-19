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
# CONFIGURACIÓN - TUS TICKERS
# ==========================================
st.set_page_config(page_title="APEX 150K Rápido", layout="wide")

PORTFOLIO_FILE = "portfolio.json"
TARGET_GOAL = 150000
STRUCTURAL_RESERVE_PCT = 0.08
DEFAULT_MONTHLY = 400

ASSETS = {
    "BTC-EUR": "BTC-EUR",
    "EMXC.DE": "EMXC.DE",
    "IS3Q.DE": "IS3Q.DE",
    "PPFB.DE": "PPFB.DE",
    "URNU.DE": "URNU.DE",
    "VVSM.DE": "VVSM.DE",
    "ZPRR.DE": "ZPRR.DE"
}

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
# CARGA DE PORTAFOLIO
# ==========================================
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    else:
        default = {
            "positions": {t: {"shares": 0.0, "avg_price": 0.0} for t in ASSETS},
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
# PRECIOS ACTUALES (SOLO 1 DÍA) - MUY RÁPIDO
# ==========================================
@st.cache_data(ttl=300)
def get_current_prices():
    tickers = list(ASSETS.values()) + ["^VIX", "^TNX", "^IRX", "^GSPC"]
    try:
        # Descargar solo 1 día
        data = yf.download(tickers, period="1d", progress=False)["Close"].iloc[-1]
        prices = {name: data[ticker] for name, ticker in ASSETS.items()}
        macro = {k: data[k] for k in ["^VIX", "^TNX", "^IRX", "^GSPC"]}
        return prices, macro
    except Exception as e:
        st.error(f"Error al obtener precios: {e}")
        return None, None

prices, macro = get_current_prices()
if prices is None:
    st.stop()

# ==========================================
# CÁLCULOS INICIALES (SIN HISTÓRICOS)
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

# ==========================================
# INTERFAZ PRINCIPAL (CARGA INMEDIATA)
# ==========================================
st.title("🚀 **APEX 150K ELITE — Versión Rápida**")
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

# Métricas rápidas
col1, col2, col3, col4 = st.columns(4)
col1.metric("BTC Precio", f"{prices['BTC-EUR']:,.0f} €")
col2.metric("VIX", f"{macro['^VIX']:.1f}")
col3.metric("Reserva", f"{portfolio['cash_reserve']:.2f} €")
col4.metric("Valor Cartera", f"{total_value:,.0f} €")

st.markdown("---")

# Tabla de cartera (rápida)
df = pd.DataFrame([
    [name,
     portfolio["positions"][name]["shares"],
     prices[name],
     current_values[name]]
    for name in ASSETS
], columns=["Activo", "Shares", "Precio", "Valor"])
st.dataframe(df.style.format({"Shares": "{:.4f}", "Precio": "{:.2f}", "Valor": "{:.2f}"}))

st.markdown("---")

# ==========================================
# SECCIÓN DE CÁLCULOS PESADOS (BAJO DEMANDA)
# ==========================================
st.subheader("📈 Análisis Avanzado")

if st.button("🔍 Calcular optimización y Monte Carlo"):
    with st.spinner("Descargando datos históricos (1 año) y calculando... (puede tomar hasta 30 segundos)"):
        # Descargar datos históricos (solo 1 año para agilizar)
        try:
            hist = yf.download(list(ASSETS.values()), period="1y", progress=False)["Close"].ffill().bfill()
            hist.columns = list(ASSETS.keys())
            returns = hist.pct_change().dropna()
        except Exception as e:
            st.error(f"Error en datos históricos: {e}")
            st.stop()

        # Z-score BTC (necesita 200 días, con 1 año es suficiente)
        btc_series = hist["BTC-EUR"]
        ma200 = btc_series.rolling(200).mean()
        std200 = btc_series.rolling(200).std()
        btc_z = (btc_series.iloc[-1] - ma200.iloc[-1]) / std200.iloc[-1]

        # Régimen con VIX (necesitamos percentiles, usamos los últimos 2 años de VIX)
        vix_hist = yf.download("^VIX", period="2y", progress=False)["Close"]
        vix_p80 = vix_hist.quantile(0.8)
        vix_p20 = vix_hist.quantile(0.2)
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
        if btc_z < -2:
            regime = "ATTACK_MODE"
            target_vol = 0.22

        # Optimización
        mu = returns.mean() * 252
        cov = returns.cov() * 252

        def optimize(btc_min, btc_max):
            n = len(ASSETS)
            names = list(ASSETS.keys())
            def neg_sharpe(w):
                return - (w @ mu) / np.sqrt(w @ cov @ w)
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w)-1},
                           {'type': 'ineq', 'fun': lambda w: target_vol - np.sqrt(w @ cov @ w)}]
            for sector in set(SECTOR_MAP.values()):
                idx = [i for i, name in enumerate(names) if SECTOR_MAP[name] == sector]
                if idx:
                    constraints.append({'type': 'ineq', 'fun': lambda w, i=idx: SECTOR_CAP - np.sum(w[i])})
            bounds = [(0.02, 0.4)] * n
            bounds[names.index("BTC-EUR")] = (btc_min, btc_max)
            res = minimize(neg_sharpe, np.ones(n)/n, bounds=bounds, constraints=constraints, method='SLSQP')
            return pd.Series(res.x, index=names)

        target_weights = optimize(btc_min, btc_max)
        expected_return = target_weights @ mu

        # Monte Carlo
        def mc(val, years, mu_ann, vol_ann, n=500):
            months = years*12
            monthly_mu = mu_ann/12
            monthly_vol = vol_ann/np.sqrt(12)
            res = []
            for _ in range(n):
                v = val
                for m in range(months):
                    v = v * (1 + np.random.normal(monthly_mu, monthly_vol)) + monthly_injection
                res.append(v)
            return np.array(res)

        base = mc(total_value, 10, expected_return, target_vol)
        prob_base = np.mean(base >= TARGET_GOAL)

        # Mostrar resultados
        st.success(f"Probabilidad 150K: {prob_base:.1%}")
        st.metric("Régimen", regime)

        # Donut objetivo
        st.plotly_chart(px.pie(names=target_weights.index, values=target_weights.values, hole=0.6))

        # Riesgo
        risk = (current_weights * (cov @ current_weights)) / np.sqrt(current_weights @ cov @ current_weights)
        risk = risk / risk.sum()
        st.plotly_chart(px.pie(names=target_weights.index, values=risk, hole=0.6))

        # Histograma Monte Carlo
        fig = go.Figure(data=[go.Histogram(x=base, nbinsx=40)])
        fig.add_vline(x=TARGET_GOAL, line_dash="dash", line_color="red")
        st.plotly_chart(fig)

# ==========================================
# APORTE MANUAL Y ÓRDENES (rápidos)
# ==========================================
st.markdown("---")
st.subheader("💰 Aporte manual")
aporte = st.number_input("Cantidad (€)", 0.0, step=50.0, key="aporte")
if st.button("Añadir a reserva"):
    portfolio["cash_reserve"] += aporte
    portfolio["last_updated"] = datetime.now().isoformat()
    save_portfolio(portfolio)
    st.success("Reserva actualizada. Recarga para ver cambios.")

st.markdown("---")
st.caption("Para análisis avanzado, pulsa el botón de arriba. Los precios se actualizan cada 5 minutos.")
