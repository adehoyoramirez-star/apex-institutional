import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import plotly.express as px
import requests
from scipy.optimize import minimize

st.set_page_config(layout="wide", page_title="APEX Institutional")

# =========================
# CONFIG (Respetando tu portfolio.json)
# =========================
with open("portfolio.json") as f:
    config = json.load(f)

positions = config["positions"]
monthly_injection = config["monthly_injection"]
btc_cap = config["btc_cap"]
target_goal = config["target_goal"]
telegram_token = config.get("telegram_token")
telegram_chat = config.get("telegram_chat_id")
tickers = list(positions.keys())

def send_telegram(msg):
    if telegram_token and telegram_chat:
        url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
        try:
            requests.post(url, data={"chat_id": telegram_chat, "text": msg, "parse_mode": "Markdown"})
        except:
            pass

# =========================
# DOWNLOAD DATA
# =========================
all_tickers = tickers + ["^VIX", "^TNX", "^GSPC"]
raw_data = yf.download(all_tickers, period="5y", auto_adjust=True)["Close"].ffill()

data = raw_data[tickers].ffill()
returns = data.pct_change().dropna()
latest_prices = data.iloc[-1]

# =========================
# INDICADORES (ERP, VIX, RSI)
# =========================
bond_10y = raw_data["^TNX"].iloc[-1] / 100
erp_real = (1 / 24.5) - bond_10y 

vix_now = float(raw_data["^VIX"].iloc[-1])
vix_p70 = float(raw_data["^VIX"].quantile(0.7))
vix_p30 = float(raw_data["^VIX"].quantile(0.3))

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
if vix_now > vix_p70: regime, target_vol = "RISK_OFF", 0.10
elif vix_now < vix_p30: regime, target_vol = "RISK_ON", 0.22
if btc_z < -2: regime, target_vol, attack_mode = "ATTACK_BTC", 0.25, True

# =========================
# OPTIMIZADOR
# =========================
cov = returns.cov() * 252
mu = returns.mean() * 252
def port_vol(w): return np.sqrt(w.T @ cov @ w)
def objective(w): return -(w @ mu)

n = len(tickers)
bounds = [(0.02, 0.45) for _ in tickers]
idx_btc = tickers.index("BTC-EUR")
bounds[idx_btc] = (0.02, 0.35 if attack_mode else btc_cap)

res = minimize(objective, np.ones(n)/n, bounds=bounds, 
               constraints=[{'type':'eq','fun': lambda w: np.sum(w)-1},
                            {'type':'ineq','fun': lambda w: target_vol - port_vol(w)}])

# PESOS ORDENADOS PARA EVITAR CRUCE DE DATOS
optimal_weights = pd.Series(res.x, index=tickers)

# =========================
# MONTE CARLO (Probabilidad e Histograma)
# =========================
total_value = sum(positions[t]["shares"] * latest_prices[t] for t in tickers)
mu_p, vol_p = optimal_weights @ mu, port_vol(optimal_weights)
sims = 10000
mc = (total_value + (monthly_injection * 120)) * np.exp((mu_p - 0.5 * vol_p**2) + vol_p * np.random.normal(0, 1, sims))
prob_goal = np.mean(mc >= target_goal)

# =================

