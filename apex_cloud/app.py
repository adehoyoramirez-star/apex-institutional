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

# ================== CONFIGURACIÓN ==================
st.set_page_config(layout="wide", page_title="APEX 150K ELITE")

PORTFOLIO_FILE = "portfolio.json"
TARGET_GOAL = 150000
STRUCTURAL_RESERVE_PCT = 0.08
DEFAULT_MONTHLY = 400

# Lista de tickers ACTUALIZADA (URNU.DE es el correcto)
TICKERS = ["BTC-EUR", "EMXC.DE", "IS3Q.DE", "PPFB.DE", "URNU.DE", "VVSM.DE", "ZPRR.DE"]

# Mapeo sectorial
SECTOR_MAP = {
    "BTC-EUR": "crypto",
    "EMXC.DE": "emerging",
    "IS3Q.DE": "global_quality",
    "PPFB.DE": "gold",
    "URNU.DE": "uranium",
    "VVSM.DE": "semis",
    "ZPRR.DE": "smallcap_usa"
}

SECTOR_CAP = 0.35  # Límite por sector

# ================== FUNCIONES DE PERSISTENCIA ==================
def load_portfolio():
    """Carga cartera desde JSON con manejo de errores."""
    if not os.path.exists(PORTFOLIO_FILE):
        # Crear archivo por defecto
        default = {
            "positions": {t: {"shares": 0, "avg_price": 0} for t in TICKERS},
            "cash_reserve": 150,  # valor inicial típico
            "last_updated": datetime.now().isoformat()
        }
        save_portfolio(default)
        return default

    try:
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        # Mostrar error detallado en la interfaz
        st.error(f"**Error en el archivo `portfolio.json`:** {str(e)}")
        st.markdown("""
        El archivo no tiene un formato JSON válido. Por favor, revísalo o restablece los valores por defecto.

        **Opciones:**
        - [Ver contenido actual](#) (desplegado abajo) para identificar el error.
        - **Restablecer a valores por defecto** (perderás tus datos actuales, pero podrás introducirlos de nuevo).
        """)

        # Mostrar el contenido del archivo para depuración
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            raw_content = f.read()
        st.code(raw_content, language="json")

        if st.button("⚠️ Restablecer a valores por defecto"):
            default = {
                "positions": {t: {"shares": 0, "avg_price": 0} for t in TICKERS},
                "cash_reserve": 150,
                "last_updated": datetime.now().isoformat()
            }
            save_portfolio(default)
            st.rerun()
        st.stop()  # No continuar hasta que se solucione

    # Asegurar que todas las claves existen
    for t in TICKERS:
        if t not in data["positions"]:
            data["positions"][t] = {"shares": 0, "avg_price": 0}
    return data

def save_portfolio(portfolio):
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2, ensure_ascii=False)

# ================== RESTO DEL CÓDIGO (sin cambios) ==================
# ... (aquí va todo el código que tenías antes, desde get_market_data hasta el final)
# Por brevedad, no repito todo, pero debes mantenerlo igual que en la versión anterior.
# Asegúrate de que en la parte de optimización y demás no haya referencias a U3O8.DE, sino a URNU.DE.
