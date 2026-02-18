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
# DOWNLOAD DATA (Añadimos Bonos y S&P500 para ERP)
# =========================
all_tickers = tickers + ["^VIX", "^TNX", "^GSPC"]
raw_data = yf.download(all_tickers, period="5y", auto_adjust=True)["Close"].ffill()

data = raw_data[tickers]
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
# CÁLCULO ERP REAL E INDICADORES
# =========================
# 1. ERP REAL: (1/PER S&P500) - Bono 10Y
bond_10y = raw_data["^TNX"].iloc[-1] / 100
# Usamos un PER real de mercado actual (aprox 24.5 para el S&P500 en 2026)
earnings_yield = 1 / 24.5 
erp_real = earnings_yield - bond_10y

# 2. VIX
vix_series = raw_data["^VIX"]
vix_now = float(vix_series.iloc[-1])
vix_p70 = float(vix_series.quantile(0.7))
vix_p30 = float(vix_series.quantile(0.3))

# =========================
# REGIMEN MERCADO
# =========================
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
# OPTIMIZADOR (Lógica Intacta)
# =========================
cov = returns.cov() * 252
mu = returns.mean() * 252

def port_vol(w): return np.sqrt(w.T @ cov @ w)
def objective(w): return -(w @ mu)

n = len(tickers)
bounds = [(0.02, 0.45) for _ in tickers]
btc_index = tickers.index("BTC-EUR")
bounds[btc_index] = (0.02, btc_cap if not attack_mode else 0.35)

constraints = [
    {'type':'eq','fun': lambda w: np.sum(w)-1},
    {'type':'ineq','fun': lambda w: target_vol - port_vol(w)}
]

res = minimize(objective, np.ones(n)/n, bounds=bounds, constraints=constraints)
optimal_weights = pd.Series(res.x, index=tickers)

# =========================
# MONTE CARLO
# =========================
def monte_carlo():
    mu_p, vol_p = optimal_weights @ mu, port_vol(optimal_weights)
    sims, months = 10000, 120
    results = []
    for _ in range(sims):
        v = total_value
        for m in range(months):
            v = (v + monthly_injection) * (1 + np.random.normal(mu_p/12, vol_p/np.sqrt(12)))
        results.append(v)
    return np.array(results)

mc = monte_carlo()
prob_goal = np.mean(mc >= target_goal)

# =========================
# GESTIÓN DE RESERVA Y ÓRDENES
# =========================
reserve_file = "cash_reserve.csv"
try:
    current_reserve = float(pd.read_csv(reserve_file)["reserve"].iloc[-1])
except:
    current_reserve = 0.0

total_cash = monthly_injection + current_reserve
orders, spent = {}, 0

for t in tickers:
    price = latest_prices[t]
    allocation = optimal_weights[t] * total_cash
    if t == "BTC-EUR":
        units = round(allocation / price, 6) # Decimales para BTC
        orders[t] = units
        spent += units * price
    else:
        units = int(allocation // price) # Enteros para el resto
        if units > 0:
            orders[t] = units
            spent += units * price

new_reserve = total_cash - spent
pd.DataFrame({"reserve": [new_reserve]}).to_csv(reserve_file, index=False)

# =========================
# DASHBOARD VISUAL
# =========================
st.title("🦅 APEX INSTITUTIONAL DEFINITIVE")

# FILA 1: KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cartera", f"{total_value:,.0f}€")
c2.metric("Régimen", regime)
c3.metric("ERP Real", f"{erp_real:.2%}")
c4.metric("Reserva", f"{new_reserve:.2f}€")

# FILA 2: GRÁFICOS
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🎯 Pesos Objetivo (Donut)")
    fig_donut = px.pie(names=optimal_weights.index, values=optimal_weights.values, hole=0.5)
    st.plotly_chart(fig_donut, use_container_width=True)

with col_right:
    st.subheader("🛒 Órdenes del Mes")
    st.json(orders)
    st.write(f"**VIX:** {vix_now:.2f} | **BTC Z-Score:** {btc_z:.2f}")

# FILA 3: MONTE CARLO
st.subheader("Simulación Monte Carlo (10 años)")
fig_mc = px.histogram(mc, nbins=50, title="Distribución de Valor Final")
st.plotly_chart(fig_mc, use_container_width=True)

# TELEGRAM
if st.button("Enviar Alerta"):
    msg = f"🚀 APEX: {regime}\nERP: {erp_real:.2%}\nVIX: {vix_now:.2f}\n\nCompras:\n{orders}"
    send_telegram(msg)
