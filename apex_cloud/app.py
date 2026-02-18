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
# CONFIG
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
# INDICADORES (VIX, ERP, BTC RSI)
# =========================
bond_10y = raw_data["^TNX"].iloc[-1] / 100
erp_real = (1 / 24.5) - bond_10y 

vix_series = raw_data["^VIX"]
vix_now = float(vix_series.iloc[-1])
vix_p70 = float(vix_series.quantile(0.7))
vix_p30 = float(vix_series.quantile(0.3))

btc_series = data["BTC-EUR"]
ma200 = btc_series.rolling(200).mean()
std200 = btc_series.rolling(200).std()
btc_z = (btc_series.iloc[-1] - ma200.iloc[-1]) / std200.iloc[-1]

# Cálculo RSI 14
delta = btc_series.diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
btc_rsi = 100 - (100 / (1 + rs.iloc[-1]))

# DETERMINACIÓN DE RÉGIMEN (CORREGIDO)
regime, target_vol, attack_mode = "NEUTRAL", 0.15, False

if vix_now > vix_p70:
    regime, target_vol = "RISK_OFF", 0.10
elif vix_now < vix_p30:
    regime, target_vol = "RISK_ON", 0.22

if btc_z < -2:
    # AQUÍ ESTABA EL ERROR: Faltaba el True al final
    regime, target_vol, attack_mode = "ATTACK_BTC", 0.25, True

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
optimal_weights = pd.Series(res.x, index=tickers)

# =========================
# MONTE CARLO (PROBABILIDAD REAL)
# =========================
total_value = sum(positions[t]["shares"] * latest_prices[t] for t in tickers)
mu_p, vol_p = optimal_weights @ mu, port_vol(optimal_weights)

# Simulación simplificada Log-Normal para evitar Prob 0
sims = 10000
# Proyectamos capital final con aportaciones
mc_final = (total_value + (monthly_injection * 120)) * np.exp((mu_p - 0.5 * vol_p**2) + vol_p * np.random.normal(0, 1, sims))
prob_goal = np.mean(mc_final >= target_goal)

# =========================
# RESERVA Y ÓRDENES
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
    if pd.isna(price) or price <= 0: continue
    
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
# DASHBOARD VISUAL
# =========================
st.title("🦅 APEX INSTITUTIONAL")

# KPIs GRANDES (Móvil friendly)
m1, m2, m3, m4 = st.columns(4)
m1.metric("RÉGIMEN", regime)
m2.metric("PROB. ÉXITO", f"{prob_goal:.1%}")
m3.metric("BTC RSI", f"{btc_rsi:.2f}")
m4.metric("ERP REAL", f"{erp_real:.2%}")

st.divider()

# ANÁLISIS EXPLICATIVO
col_e1, col_e2 = st.columns(2)
with col_e1:
    if attack_mode:
        st.error(f"⚠️ **MODO ATAQUE:** BTC en zona de compra extrema (Z-Score: {btc_z:.2f}).")
    elif btc_rsi < 35:
        st.warning(f"📉 **SOBREVENTA:** BTC RSI en {btc_rsi:.2f}. Oportunidad técnica.")
    else:
        st.success(f"✅ **SITUACIÓN:** Operativa normal bajo régimen {regime}.")

with col_e2:
    st.info(f"**Indicadores:** VIX ({vix_now:.2f}) / ERP ({erp_real:.2%}). El sistema está invirtiendo {total_cash:.2f}€ este mes.")

# GRÁFICOS
c_l, c_r = st.columns(2)
with c_l:
    st.subheader("🎯 Pesos Objetivo (Donut)")
    # Corregido: Mostrar los pesos óptimos correctamente en el Donut
    fig = px.pie(names=optimal_weights.index, values=optimal_weights.values, hole=0.6,
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with c_r:
    st.subheader("🛒 Órdenes de Compra")
    st.json(orders)
    st.write(f"**Pólvora sobrante:** {new_reserve:.2f}€")

# TELEGRAM
if st.button("🚀 Enviar Informe"):
    msg = f"🦅 *APEX ALERTA*\n\n*Régimen:* {regime}\n*Prob:* {prob_goal:.1%}\n*BTC RSI:* {btc_rsi:.2f}\n*ERP:* {erp_real:.2%}\n\n*Compras:* `{orders}`"
    send_telegram(msg)
    st.toast("Informe enviado")
