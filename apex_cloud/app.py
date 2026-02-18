import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
import requests

st.set_page_config(layout="wide", page_title="APEX 150K ELITE")

# =========================
# CONFIG BASE
# =========================

CONFIG = {
    "monthly_injection": 400,
    "cash_reserve": 150,
    "target_goal": 150000,
    "structural_reserve_pct": 0.08,
    "btc_cap": 0.30,
    "telegram_token": "",
    "telegram_chat_id": "",
    "positions": {
        "BTC-EUR": {"shares": 0.05},
        "EMXC.DE": {"shares": 5},
        "IS3Q.DE": {"shares": 3},
        "PPFB.DE": {"shares": 2},
        "U3O8.DE": {"shares": 4},
        "VVSM.DE": {"shares": 1},
        "ZPRR.DE": {"shares": 2}
    }
}

monthly_injection = CONFIG["monthly_injection"]
cash_reserve = CONFIG["cash_reserve"]
target_goal = CONFIG["target_goal"]
reserve_pct = CONFIG["structural_reserve_pct"]
btc_cap = CONFIG["btc_cap"]
positions = CONFIG["positions"]

tickers = list(positions.keys())

# =========================
# DATA
# =========================

all_tickers = tickers + ["^VIX", "^TNX", "^GSPC"]
raw = yf.download(all_tickers, period="5y", auto_adjust=True)["Close"].ffill()

data = raw[tickers]
returns = data.pct_change().dropna()
latest_prices = data.iloc[-1]

btc_price = latest_prices["BTC-EUR"]
vix = raw["^VIX"].iloc[-1]
sp500 = raw["^GSPC"]

# =========================
# RÉGIMEN + ATAQUE
# =========================

vix_p80 = raw["^VIX"].quantile(0.8)
vix_p20 = raw["^VIX"].quantile(0.2)

regime = "NEUTRAL"
target_vol = 0.16
attack = False

if vix > vix_p80:
    regime = "RISK_OFF"
    target_vol = 0.10
elif vix < vix_p20:
    regime = "RISK_ON"
    target_vol = 0.22

# Ataque extremo BTC (-2σ)
btc_series = data["BTC-EUR"]
ma200 = btc_series.rolling(200).mean()
std200 = btc_series.rolling(200).std()
btc_z = (btc_series.iloc[-1] - ma200.iloc[-1]) / std200.iloc[-1]

if btc_z < -2:
    attack = True
    regime = "ATTACK_MODE"

# =========================
# OPTIMIZACIÓN
# =========================

cov = returns.cov() * 12
mu = returns.mean() * 12

def port_vol(w):
    return np.sqrt(w.T @ cov @ w)

def objective(w):
    return -(w @ mu)

n = len(tickers)
bounds = [(0.02, 0.40) for _ in range(n)]
btc_idx = tickers.index("BTC-EUR")
bounds[btc_idx] = (0.02, btc_cap if not attack else 0.40)

res = minimize(
    objective,
    np.ones(n)/n,
    bounds=bounds,
    constraints=[
        {'type':'eq','fun': lambda w: np.sum(w)-1},
        {'type':'ineq','fun': lambda w: target_vol - port_vol(w)}
    ]
)

weights = pd.Series(res.x, index=tickers)

# =========================
# CARTERA ACTUAL
# =========================

current_values = {
    t: positions[t]["shares"] * latest_prices[t]
    for t in tickers
}

current_total = sum(current_values.values())
current_weights = pd.Series(current_values) / current_total

# =========================
# MONTE CARLO MENSUAL REAL
# =========================

years = 10
months = years * 12
sims = 5000

mu_p = weights @ mu
vol_p = port_vol(weights)

monthly_mu = mu_p / 12
monthly_vol = vol_p / np.sqrt(12)

mc_results = []

for _ in range(sims):
    value = current_total
    for m in range(months):
        shock = np.random.normal(monthly_mu, monthly_vol)
        value = value * (1 + shock) + monthly_injection
    mc_results.append(value)

mc_results = np.array(mc_results)
prob_goal = np.mean(mc_results >= target_goal)

# =========================
# RESERVA INTELIGENTE
# =========================

structural_reserve = reserve_pct * (current_total + monthly_injection)
usable_cash = max(0, (cash_reserve + monthly_injection) - structural_reserve)

if attack:
    usable_cash = cash_reserve + monthly_injection

# =========================
# ÓRDENES
# =========================

orders = {}
spent = 0

for t in tickers:
    alloc = weights[t] * usable_cash
    price = latest_prices[t]

    if t == "BTC-EUR":
        units = round(alloc / price, 6)
        orders[t] = units
        spent += units * price
    else:
        units = int(alloc // price)
        if units > 0:
            orders[t] = units
            spent += units * price

remaining_reserve = cash_reserve + monthly_injection - spent

# =========================
# DASHBOARD
# =========================

st.title("🦅 APEX 150K ELITE")

c1, c2, c3, c4 = st.columns(4)
c1.metric("RÉGIMEN", regime)
c2.metric("BTC Precio", f"{btc_price:,.0f} €")
c3.metric("Probabilidad 150K", f"{prob_goal:.1%}")
c4.metric("Reserva Actual", f"{remaining_reserve:.2f} €")

st.divider()

# Donut Objetivo
st.subheader("🎯 Objetivo vs Actual")

df_compare = pd.DataFrame({
    "Objetivo": weights,
    "Actual": current_weights
}).fillna(0)

fig = px.pie(
    names=df_compare.index,
    values=df_compare["Objetivo"],
    hole=0.6,
    title="Asignación Objetivo"
)
st.plotly_chart(fig, use_container_width=True)

st.write("Desviación actual:")
st.dataframe((df_compare["Actual"] - df_compare["Objetivo"]).sort_values())

# Monte Carlo
st.subheader("📈 Monte Carlo 10 años")
fig_mc = px.histogram(mc_results, nbins=50)
st.plotly_chart(fig_mc, use_container_width=True)

# Diagnóstico
st.subheader("🧠 Diagnóstico Mercado")

if regime == "RISK_ON":
    st.success("Volatilidad baja. Momentum favorable. Entorno constructivo.")
elif regime == "RISK_OFF":
    st.warning("Alta volatilidad. Riesgo elevado. Priorizando defensa.")
elif regime == "ATTACK_MODE":
    st.error("Capitulación extrema detectada. Activando modo ataque.")
else:
    st.info("Mercado neutral. Posicionamiento equilibrado.")

# Compras
st.subheader("🛒 Qué Comprar y Por Qué")

for t in orders:
    reason = "Alineación a peso óptimo."
    if attack and t == "BTC-EUR":
        reason = "Evento extremo BTC detectado. Ataque activado."
    st.write(f"• **{t}** → {orders[t]} unidades | {reason}")

st.write(f"Reserva estructural protegida: {structural_reserve:.2f} €")
st.write(f"Reserva restante tras compras: {remaining_reserve:.2f} €")
