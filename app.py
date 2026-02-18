import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import plotly.express as px
import requests
from scipy.optimize import minimize

st.set_page_config(layout="wide")

# =========================
# CONFIG
# =========================
with open("portfolio.json") as f:
    config = json.load(f)

positions = config["positions"]
monthly_injection = config["monthly_injection"]
btc_cap = config["btc_cap"]
max_dd = config["max_drawdown"]
target_goal = config["target_goal"]

telegram_token = config.get("telegram_token", None)
telegram_chat = config.get("telegram_chat_id", None)

tickers = list(positions.keys())

# =========================
# TELEGRAM
# =========================
def send_telegram(msg):
    if telegram_token and telegram_chat:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        requests.post(url, data={"chat_id": telegram_chat, "text": msg})

# =========================
# DOWNLOAD DATA
# =========================
data = yf.download(tickers, period="5y", auto_adjust=True)["Close"]

if data.isna().all().any():
    st.error("Algún ticker no tiene datos válidos. Revisa símbolos.")
    st.stop()

returns = data.pct_change().dropna()
latest_prices = data.iloc[-1]

# =========================
# VALOR REAL CARTERA
# =========================
values = {}
total_value = 0

for t in tickers:
    shares = positions[t]["shares"]
    value = shares * latest_prices[t]
    values[t] = value
    total_value += value

weights_current = pd.Series(values) / total_value

# =========================
# REGIMEN MERCADO (Versión Corregida para Streamlit)
# =========================
vix_data = yf.download("^VIX", period="3y", progress=False)["Close"]

# Usamos .squeeze() para asegurar que si es un DataFrame de una columna, se convierta en Serie
if isinstance(vix_data, pd.DataFrame):
    vix = vix_data.iloc[:, 0] # Tomamos la primera columna
else:
    vix = vix_data

# Extraemos los valores como números puros (floats)
vix_now = float(vix.iloc[-1])
vix_p70 = float(vix.quantile(0.7))
vix_p30 = float(vix.quantile(0.3))

# Ahora la comparación funciona perfectamente
if vix_now > vix_p70:
    regime = "RISK_OFF"
    target_vol = 0.10
elif vix_now < vix_p30:
    regime = "RISK_ON"
    target_vol = 0.22
else:
    regime = "NEUTRAL"
    target_vol = 0.15
# =========================
# BTC CAPITULATION
# =========================
btc = data["BTC-EUR"]
ma200 = btc.rolling(200).mean()
std200 = btc.rolling(200).std()

btc_z = (btc.iloc[-1] - ma200.iloc[-1]) / std200.iloc[-1]
btc_dd = (btc.iloc[-1] / btc.cummax().iloc[-1]) - 1

attack_mode = False
if btc_z < -2 and btc_dd < -0.35:
    regime = "ATTACK_BTC"
    target_vol = 0.25
    attack_mode = True

# =========================
# OPTIMIZADOR
# =========================
cov = returns.cov() * 252
mu = returns.mean() * 252

def port_vol(w):
    return np.sqrt(w.T @ cov @ w)

def objective(w):
    return -(w @ mu)

n = len(tickers)
bounds = [(0.02, 0.45) for _ in tickers]

btc_index = tickers.index("BTC-EUR")
bounds[btc_index] = (0.02, btc_cap if not attack_mode else 0.35)

constraints = [
    {'type':'eq','fun': lambda w: np.sum(w)-1},
    {'type':'ineq','fun': lambda w: target_vol - port_vol(w)}
]

w0 = np.ones(n)/n
res = minimize(objective, w0, bounds=bounds, constraints=constraints)

if not res.success:
    st.error("Optimización falló.")
    st.stop()

optimal_weights = pd.Series(res.x, index=tickers)

# =========================
# MONTE CARLO 10Y
# =========================
def monte_carlo():
    mu_p = optimal_weights @ mu
    vol_p = port_vol(optimal_weights)
    sims = 10000
    months = 120
    results = []

    for _ in range(sims):
        value = total_value
        for m in range(months):
            shock = np.random.normal(mu_p/12, vol_p/np.sqrt(12))
            value = (value + monthly_injection) * (1 + shock)
        results.append(value)

    return np.array(results)

mc = monte_carlo()
prob_goal = np.mean(mc >= target_goal)
median = np.median(mc)

# =========================
# ORDENES SIN FRACCIONES
# =========================
orders = {}
cash = monthly_injection

for t in tickers:
    price = latest_prices[t]
    allocation = optimal_weights[t] * cash
    units = int(allocation // price)
    if units > 0:
        orders[t] = units

# =========================
# RAZONES DE COMPRA
# =========================
explanation = []
if attack_mode:
    explanation.append("BTC en capitulación estadística extrema.")
if regime == "RISK_ON":
    explanation.append("Volatilidad baja. Entorno expansivo.")
if regime == "RISK_OFF":
    explanation.append("Alta volatilidad. Posicionamiento defensivo.")
if regime == "NEUTRAL":
    explanation.append("Mercado en rango intermedio.")

# =========================
# DASHBOARD
# =========================
st.title("APEX INSTITUTIONAL DEFINITIVE")

col1,col2,col3,col4 = st.columns(4)
col1.metric("Valor Cartera (€)", f"{total_value:,.0f}")
col2.metric("Régimen", regime)
col3.metric("BTC Z-score", f"{btc_z:.2f}")
col4.metric("Prob ≥150k (10Y)", f"{prob_goal:.1%}")

st.subheader("Qué comprar este mes")
st.write(orders)

st.subheader("Por qué")
for e in explanation:
    st.write("-", e)

st.subheader("Pesos óptimos")
st.dataframe(optimal_weights)

st.subheader("Monte Carlo")
fig = px.histogram(mc, nbins=60)
st.plotly_chart(fig)

# =========================
# ALERTA TELEGRAM
# =========================
message = f"""
APEX ALERTA

Regimen: {regime}
BTC Z: {btc_z:.2f}
Probabilidad 150k: {prob_goal:.1%}
Comprar: {orders}
"""
# Añade esto a tu variable 'report'
report += f"📊 VIX Actual: `{vix_now:.2f}` (Umbral Riesgo: `{vix_p70:.2f}`)"

send_telegram(message)
