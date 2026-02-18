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
target_goal = config["target_goal"]

# Acceso correcto a Telegram desde el JSON
telegram_config = config.get("telegram", {})
telegram_token = telegram_config.get("token")
telegram_chat = telegram_config.get("chat_id")

tickers = list(positions.keys())

# =========================
# TELEGRAM
# =========================
def send_telegram(msg):
    if telegram_token and telegram_chat:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        try:
            requests.post(url, data={"chat_id": telegram_chat, "text": msg, "parse_mode": "Markdown"})
        except Exception as e:
            st.error(f"Error Telegram: {e}")

# =========================
# DOWNLOAD DATA
# =========================
all_tickers = tickers + ["^VIX", "^TNX", "^GSPC"]
raw_data = yf.download(all_tickers, period="5y", auto_adjust=True)["Close"].ffill()

data = raw_data[tickers]
returns = data.pct_change().dropna()
latest_prices = data.iloc[-1]

# =========================
# VALOR REAL CARTERA
# =========================
values = {t: positions[t]["shares"] * latest_prices[t] for t in tickers}
total_value = sum(values.values())

# =========================
# INDICADORES (VIX, ERP, BTC RSI)
# =========================
# 1. ERP REAL: (1/PER S&P500) - Bono 10Y
bond_10y = raw_data["^TNX"].iloc[-1] / 100
erp_real = (1 / 24.5) - bond_10y

# 2. VIX
vix_series = raw_data["^VIX"]
vix_now = float(vix_series.iloc[-1])
vix_p70 = float(vix_series.quantile(0.7))
vix_p30 = float(vix_series.quantile(0.3))

# 3. BTC RSI Y Z-SCORE
btc_series = data["BTC-EUR"]
ma200 = btc_series.rolling(200).mean()
std200 = btc_series.rolling(200).std()
btc_z = (btc_series.iloc[-1] - ma200.iloc[-1]) / std200.iloc[-1]

delta = btc_series.diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
btc_rsi = 100 - (100 / (1 + rs.iloc[-1]))

# RÉGIMEN
regime, target_vol, attack_mode = "NEUTRAL", 0.15, False
if vix_now > vix_p70:
    regime, target_vol = "RISK_OFF", 0.10
elif vix_now < vix_p30:
    regime, target_vol = "RISK_ON", 0.22

if btc_z < -2:
    regime, target_vol, attack_mode = "ATTACK_BTC", 0.25, True

# =========================
# OPTIMIZADOR
# =========================
cov = returns.cov() * 252
mu = returns.mean() * 252

def port_vol(w): return np.sqrt(w.T @ cov @ w)
def objective(w): return -(w @ mu)

bounds = [(0.02, 0.45) for _ in tickers]
btc_idx = tickers.index("BTC-EUR")
bounds[btc_idx] = (0.02, 0.35 if attack_mode else btc_cap)

res = minimize(objective, np.ones(len(tickers))/len(tickers), bounds=bounds, 
               constraints=[{'type':'eq','fun': lambda w: np.sum(w)-1},
                            {'type':'ineq','fun': lambda w: target_vol - port_vol(w)}])
optimal_weights = pd.Series(res.x, index=tickers)

# =========================
# MONTE CARLO (Simulación Log-Normal)
# =========================
mu_p, vol_p = optimal_weights @ mu, port_vol(optimal_weights)
sims = 10000
# Proyección a 10 años (120 meses) considerando inyección mensual
mc = (total_value + (monthly_injection * 120)) * np.exp((mu_p - 0.5 * vol_p**2) + vol_p * np.random.normal(0, 1, sims))
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
    alloc = optimal_weights[t] * total_cash
    if t == "BTC-EUR":
        units = round(alloc / price, 6)
        orders[t] = units
        spent += units * price
    else:
        units = int(alloc // price)
        if units > 0:
            orders[t] = units
            spent += units * price

new_reserve = total_cash - spent
pd.DataFrame({"reserve": [new_reserve]}).to_csv(reserve_file, index=False)

# =========================
# DASHBOARD
# =========================
st.title("🦅 APEX INSTITUTIONAL DEFINITIVE")

# KPIs Gigantes
c1, c2, c3, c4 = st.columns(4)
c1.metric("RÉGIMEN", regime)
c2.metric("PROB. ÉXITO", f"{prob_goal:.1%}")
c3.metric("BTC RSI", f"{btc_rsi:.2f}")
c4.metric("ERP REAL", f"{erp_real:.2%}")

st.divider()

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("🎯 Pesos Objetivo (Donut)")
    # El Donut ahora usa los pesos ÓPTIMOS del optimizador, no los actuales
    fig_donut = px.pie(names=optimal_weights.index, values=optimal_weights.values, hole=0.5)
    st.plotly_chart(fig_donut, use_container_width=True)

with col_r:
    st.subheader("🛒 Órdenes de Compra")
    st.json(orders)
    st.write(f"**Pólvora sobrante:** {new_reserve:.2f}€")

st.subheader("📈 Proyección Monte Carlo (10Y)")
fig_mc = px.histogram(mc, nbins=50, title="
