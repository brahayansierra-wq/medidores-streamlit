import streamlit as st
import pandas as pd

st.set_page_config(page_title="Monitoreo de medidores", layout="wide")

st.title("📊 Sistema de monitoreo metrológico de medidores de agua")
st.write(
    """
    Esta es una **versión inicial** de la aplicación del trabajo de grado.
    Aquí se integrarán:
    - Modelos de intervalo de error EQ3
    - Modelos de clasificación de conformidad
    - Estimación de vida útil remanente por modelo
    """
)

st.subheader("Panel de prueba")
st.write("Si ves esta página en Streamlit Cloud, ¡el despliegue básico está funcionando! 🚀")
