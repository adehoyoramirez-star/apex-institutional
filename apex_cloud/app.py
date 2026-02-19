# ==========================================
# QUANTUM-PLUS BALANCED ACTIVA
# Institutional Portfolio Engine vFinal
# 100% Ejecutable - Streamlit
# ==========================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# CONFIG
# ==========================================

st.set_page_config(page_title="Quantum-Plus Engine", layout="wide")

PORTFOLIO_FILE = "portfolio_state.json"

TICKERS = {
    "MSCI World Quality": "IWQU.AS",
    "Emerging Markets IMI": "IS3N.AS",
    "Global Aggregate Bond": "AGGH.AS",
    "WisdomTree AI ETF": "WTAI.AS",
    "Physical Gold": "SGLN.AS",
    "Uranium ETF": "URNM",
    "Bitcoin": "BTC-USD"
}

# ==========================================
# LOAD / SAVE PORTFOLIO
# ==========================================

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    else:
        portfolio = {
            "cash_reserve": 150.0,
            "positions": {}
        }
        for name in TICKERS:
            portfolio["positions"][name] = {
                "shares": 0.0,
                "avg_price": 0.0
            }
        save_portfolio(portfolio)
        return portfolio

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio, f, indent=4)

portfolio = load_portfolio()

# ==========================================
# PRICE DATA
# ==========================================

@st.cache_data(ttl=300)
def get_prices():
    data = {}
    for name, ticker in TICKERS.items():
        try:
            price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
            data[name] = price
        except:
            data[name] = 0.0
    return data

prices = get_prices()

# ==========================================
# CALCULATE PORTFOLIO VALUE
# ==========================================

def calculate_portfolio_value():
    total = portfolio["cash_reserve"]
    rows = []

    for name in TICKERS:
        shares = portfolio["positions"][name]["shares"]
        avg_price = portfolio["positions"][name]["avg_price"]
        current_price = prices[name]
        value = shares * current_price
        total += value

        rows.append([
            name,
            shares,
            avg_price,
            current_price,
            value
        ])

    df = pd.DataFrame(rows, columns=[
        "Activo", "Shares", "Precio Medio", "Precio Actual", "Valor Actual"
    ])

    return total, df

total_value, df_portfolio = calculate_portfolio_value()

# ==========================================
# HEADER
# ==========================================

st.title("Quantum-Plus Balanced Activa")
st.subheader("Institutional Portfolio Engine")

col1, col2 = st.columns(2)
col1.metric("Valor Total Cartera (€)", f"{total_value:,.2f}")
col2.metric("Reserva Disponible (€)", f"{portfolio['cash_reserve']:,.2f}")

st.dataframe(df_portfolio, use_container_width=True)

# ==========================================
# APORTE MENSUAL
# ==========================================

st.markdown("## Aporte de Capital")

aporte = st.number_input("Introduce aporte (€)", min_value=0.0, step=50.0)

if st.button("Añadir a Reserva"):
    portfolio["cash_reserve"] += aporte
    save_portfolio(portfolio)
    st.success("Reserva actualizada")
    st.rerun()

# ==========================================
# EJECUTAR COMPRA
# ==========================================

st.markdown("## Ejecutar Compra")

activo = st.selectbox("Selecciona Activo", list(TICKERS.keys()))
monto = st.number_input("Monto a invertir (€)", min_value=0.0, step=50.0)

if st.button("Confirmar Ejecución"):
    if monto <= portfolio["cash_reserve"] and monto > 0:
        price = prices[activo]
        shares_bought = monto / price

        old_shares = portfolio["positions"][activo]["shares"]
        old_avg = portfolio["positions"][activo]["avg_price"]

        new_total_shares = old_shares + shares_bought
        new_avg_price = (
            (old_shares * old_avg + shares_bought * price)
            / new_total_shares
        )

        portfolio["positions"][activo]["shares"] = new_total_shares
        portfolio["positions"][activo]["avg_price"] = new_avg_price
        portfolio["cash_reserve"] -= monto

        save_portfolio(portfolio)
        st.success("Compra ejecutada y cartera actualizada")
        st.rerun()
    else:
        st.error("Reserva insuficiente")

# ==========================================
# ASSET ALLOCATION CHART
# ==========================================

values = []
labels = []

for name in TICKERS:
    value = portfolio["positions"][name]["shares"] * prices[name]
    if value > 0:
        labels.append(name)
        values.append(value)

if portfolio["cash_reserve"] > 0:
    labels.append("Cash")
    values.append(portfolio["cash_reserve"])

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
fig.update_layout(title="Asset Allocation Actual")

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MONTE CARLO SIMULATION
# ==========================================

st.markdown("## Monte Carlo 5 años")

if st.button("Simular"):
    hist_data = yf.download(list(TICKERS.values()), period="5y")["Close"]
    hist_returns = hist_data.pct_change().dropna()

    weights = []
    total_invested = total_value - portfolio["cash_reserve"]

    for name in TICKERS:
        val = portfolio["positions"][name]["shares"] * prices[name]
        weight = val / total_invested if total_invested > 0 else 0
        weights.append(weight)

    weights = np.array(weights)

    mean_returns = hist_returns.mean().values
    cov_matrix = hist_returns.cov().values

    simulations = 500
    days = 252 * 5
    results = []

    for _ in range(simulations):
        daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, days)
        portfolio_path = np.cumprod(1 + np.dot(daily_returns, weights))
        results.append(portfolio_path[-1])

    fig_mc = go.Figure()
    fig_mc.add_histogram(x=results, nbinsx=40)
    fig_mc.update_layout(title="Distribución Valor Final (5 años)")

    st.plotly_chart(fig_mc, use_container_width=True)
