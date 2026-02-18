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
        requests.post(url, data={"chat_id": telegram_chat, "text": msg, "parse_mode": "Markdown"})

# =========================
# DOWNLOAD DATA
# =========================
all_tickers = tickers + ["^VIX", "^TNX", "^GSPC"]
raw_data = yf.download(all_tickers, period="5y", auto_adjust=True)["Close"].ffill()

data = raw_data[tickers]
returns = data.pct_change().dropna()
latest_prices = data.iloc[-1]

# =========================
# VALOR REAL Y ESTADO ACTUAL
# =========================
values = {t: positions[t]["shares"] * latest_prices[t] for t in tickers}
total_value = sum(values.values())

# =========================
# INDICADORES MACRO (GRANDES)
# =========================
vix_series = raw_data["^VIX"]
vix_now = float(vix_series.iloc[-1])
vix_p70 = float(vix_series.quantile(0.7))
vix_p30 = float(vix_series.quantile(0.3))

bond_10y = raw_data["^TNX"].iloc[-1] / 100
erp_real = (1 / 24.5) - bond_10y 

btc_series = data["BTC-EUR"]
btc_price = btc_series.iloc[-1]
ma200 = btc_series.rolling(200).mean()
std200 = btc_series.rolling(200).std()
btc_z = (btc_price - ma200.iloc[-1]) / std200.iloc[-1]
btc_dd = (btc_price / btc_series.cummax().iloc[-1]) - 1

# DETERMINACIÓN DE RÉGIMEN
regime, target_vol, attack_mode = "NEUTRAL", 0.15, False
if vix_now > vix_p70:
    regime, target_vol = "RISK_OFF", 0.10
elif vix_now < vix_p30:
    regime, target_vol = "RISK_ON", 0.22

if btc_z < -2 and btc_dd < -0.35:
    regime, target_vol, attack_mode = "ATTACK_BTC", 0.25, True

# =========================
# OPTIMIZADOR (Lógica Original)
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
# MONTE CARLO
# =========================
mu_p, vol_p = optimal_weights @ mu, port_vol(optimal_weights)
mc_results = np.array([ (total_value + monthly_injection*120) * (1 + np.random.normal(mu_p, vol_p)) for _ in range(10000)])
prob_goal = np.mean(mc_results >= target_goal)

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

# Corregimos asignación: Aseguramos que se use el peso óptimo sobre el total_cash
for t in tickers:
    price = latest_prices[t]
    allocation = optimal_weights[t] * total_cash
    if t == "BTC-EUR":
        units = round(allocation / price, 6)
        orders[t] = units
        spent += units * price
    else:
        units = int(allocation // price)
        if units > 0:
            orders[t] = units
            spent += units * price

new_reserve = total_cash - spent
pd.DataFrame({"reserve": [new_reserve]}).to_csv(reserve_file, index=False)

# =========================
# DASHBOARD VISUAL
# =========================
st.title("🦅 APEX INSTITUTIONAL DEFINITIVE")

# FILA 1: KPIs GRANDES
c1, c2, c3, c4 = st.columns(4)
c1.metric("RÉGIMEN", regime)
c2.metric("PROBABILIDAD ÉXITO", f"{prob_goal:.1%}")
c3.metric("ERP REAL", f"{erp_real:.2%}")
c4.metric("VIX", f"{vix_now:.2f}")

# EXPLICACIÓN DEL MERCADO
st.divider()
st.subheader("📝 Análisis de Situación")
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    if attack_mode:
        st.error(f"**ESTADO DE ALERTA:** Bitcoin está en zona de capitulación ({btc_z:.2f} Z). El sistema fuerza compra máxima.")
    elif regime == "RISK_OFF":
        st.warning("**ESTADO DEFENSIVO:** El pánico (VIX) supera el percentil 70. Protegemos capital.")
    else:
        st.success("**ESTADO EXPANSIVO:** Volatilidad controlada. Maximizando retornos.")

with col_exp2:
    st.info(f"**Qué significa esto:** Con un ERP de {erp_real:.2%}, la bolsa {'es' if erp_real > 0.02 else 'no es'} atractiva frente al bono. El sistema invertirá **{total_cash:.2f}€** este mes.")

# FILA 2: GRÁFICOS
col_l, col_r = st.columns([1, 1])

with col_l:
    st.subheader("🎯 Pesos Óptimos Calculados")
    # CORRECCIÓN DONUT: Aseguramos que lea los pesos del optimizador
    fig_donut = px.pie(
        names=optimal_weights.index, 
        values=optimal_weights.values, 
        hole=0.5,
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_donut.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_donut, use_container_width=True)

with col_r:
    st.subheader("🛒 Órdenes y Ejecución")
    st.write("Copia estas órdenes en tu broker:")
    st.json(orders)
    st.write(f"**Pólvora sobrante (reserva):** {new_reserve:.2f}€")

# TELEGRAM CON FORMATO PROFESIONAL
if st.button("🚀 Enviar Informe a Telegram"):
    msg = f"""🦅 *INFORME APEX INSTITUTIONAL*

*Régimen:* `{regime}`
*Prob. Objetivo:* `{prob_goal:.1%}`
*ERP Real:* `{erp_real:.2%}`
*VIX:* `{vix_now:.2f}`

🛒 *ÓRDENES:*
`{json.dumps(orders, indent=2)}`

💰 *RESERVA ACUMULADA:* `{new_reserve:.2f}€`"""
    send_telegram(msg)
    st.toast("Informe enviado!")
