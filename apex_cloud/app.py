import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import plotly.express as px
from scipy.optimize import minimize

st.set_page_config(layout="wide")

# =========================
# LOAD PORTFOLIO
# =========================
with open("portfolio.json") as f:
    config = json.load(f)

positions = config["positions"]
monthly_injection = config["monthly_injection"]
btc_cap = config["btc_cap"]
max_dd = config["max_drawdown"]
target_goal = config["target_goal"]

tickers = list(positions.keys())

# =========================
# DOWNLOAD DATA
# =========================
data = yf.download(tickers, period="5y", auto_adjust=True)["Close"]
returns = data.pct_change().dropna()

latest_prices = data.iloc[-1]

# =========================
# CURRENT PORTFOLIO VALUE
# =========================
values = {}
total_value = 0

for t in tickers:
    shares = positions[t]["shares"]
    value = shares * latest_prices[t]
    values[t] = value
    total_value += value

weights_current = {t: values[t]/total_value for t in tickers}

# =========================
# BTC CAPITULATION DETECTOR
# =========================
btc_series = data["BTC-EUR"]
ma200 = btc_series.rolling(200).mean()
std200 = btc_series.rolling(200).std()

btc_z = (btc_series.iloc[-1] - ma200.iloc[-1]) / std200.iloc[-1]
btc_dd = (btc_series.iloc[-1] / btc_series.cummax().iloc[-1]) - 1

attack_mode = False
if btc_z < -2 and btc_dd < -0.35:
    attack_mode = True

# =========================
# OPTIMIZER
# =========================
cov = returns.cov() * 252
mu = returns.mean() * 252

def portfolio_vol(w):
    return np.sqrt(w.T @ cov @ w)

def objective(w):
    return - (w @ mu)

n = len(tickers)
bounds = [(0.02, 0.45) for _ in tickers]

# Limitar BTC
btc_index = tickers.index("BTC-EUR")
bounds[btc_index] = (0.02, btc_cap if not attack_mode else 0.35)

constraints = (
    {'type':'eq','fun': lambda w: np.sum(w)-1}
)

w0 = np.ones(n)/n
res = minimize(objective, w0, bounds=bounds, constraints=constraints)

optimal_weights = pd.Series(res.x, index=tickers)

# =========================
# MONTE CARLO 10Y
# =========================
def monte_carlo():
    mu_p = optimal_weights @ mu
    vol_p = portfolio_vol(optimal_weights)
    months = 120
    sims = 10000
    results = []

    for _ in range(sims):
        value = total_value
        for m in range(months):
            shock = np.random.normal(mu_p/12, vol_p/np.sqrt(12))
            value = (value + monthly_injection) * (1 + shock)
        results.append(value)

    arr = np.array(results)
    return arr

mc_results = monte_carlo()

prob_goal = np.mean(mc_results >= target_goal)
median = np.median(mc_results)
p10 = np.percentile(mc_results,10)
p90 = np.percentile(mc_results,90)

# =========================
# DASHBOARD
# =========================
st.title("APEX INSTITUTIONAL CLOUD ENGINE")

col1,col2,col3 = st.columns(3)
col1.metric("Portfolio Value (€)", f"{total_value:,.0f}")
col2.metric("BTC Z-score", f"{btc_z:.2f}")
col3.metric("Prob ≥150k (10Y)", f"{prob_goal:.1%}")

st.subheader("Optimal Weights")
st.dataframe(optimal_weights)

st.subheader("Monte Carlo Distribution")
fig = px.histogram(mc_results, nbins=60)
st.plotly_chart(fig)

st.subheader("BTC Capitulation")
st.write("Attack Mode:", attack_mode)
st.write("BTC Drawdown:", f"{btc_dd:.2%}")
st.write("BTC Z-score:", f"{btc_z:.2f}")

# =========================
# NO FRACTION ORDER ENGINE
# =========================
st.subheader("Suggested Orders (No Fractions)")

orders = {}
cash = monthly_injection

for t in tickers:
    price = latest_prices[t]
    allocation = optimal_weights[t] * cash
    units = int(allocation // price)
    if units > 0:
        orders[t] = units

st.write(orders)
