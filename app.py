import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
import os

# ===========================================
# CONFIGURACIONES GENERALES
# ===========================================
EMP_Q3 = 4 - (4/3)   # ~2.67%
MAX_YEARS = 15

st.set_page_config(
    page_title="Sistema Metrológico de Medidores",
    layout="wide"
)

# ===========================================
# CARGA DE MODELOS LOCALES
# ===========================================
@st.cache_resource
def cargar_modelos():
    interval_models = {}
    clasif_models = {}

    base_path = "modelos"

    for file in os.listdir(base_path):
        if file.endswith("_lower.cbm"):
            modelo = file.replace("_lower.cbm", "")
            path = os.path.join(base_path, file)
            interval_models.setdefault(modelo, {})["lower"] = CatBoostRegressor().load_model(path)

        elif file.endswith("_upper.cbm"):
            modelo = file.replace("_upper.cbm", "")
            path = os.path.join(base_path, file)
            interval_models.setdefault(modelo, {})["upper"] = CatBoostRegressor().load_model(path)

        elif file.endswith(".cbm"): 
            modelo = file.replace(".cbm", "")
            path = os.path.join(base_path, file)
            clasif_models[modelo] = CatBoostClassifier().load_model(path)

    return interval_models, clasif_models

interval_models, clasif_models = cargar_modelos()

# ===========================================
# CARGA DE TABLA DE DEGRADACIÓN
# ===========================================
df_degrad = pd.read_csv("data/vida_util_degradacion.csv")

# ===========================================
# UI PRINCIPAL
# ===========================================
st.title("📊 Sistema de Monitoreo Metrológico de Medidores de Agua")
st.markdown("""
Esta es la versión inicial del sistema, cargando:

- Modelos de intervalo EQ3 (cuantílicos)
- Modelos de clasificación de conformidad
- Tabla de degradación por modelo
- Procesamiento local en Streamlit Cloud
""")

st.subheader("📥 Cargar datos de medidores")

archivo = st.file_uploader("Sube un archivo CSV de medidores", type=["csv"])

if archivo:
    df = pd.read_csv(archivo)
    st.success("Archivo cargado correctamente.")
    st.dataframe(df)

    # ----------------------------------------
    # BOTÓN PARA EJECUTAR PREDICCIONES
    # ----------------------------------------
    if st.button("Ejecutar predicción"):
        resultados = []

        for _, row in df.iterrows():
            modelo = str(row["Descripción"]).strip()
            edad = row["Edad"]
            volumen = row["Volumen"]
            consumo = row.get("Consumo_anual", 0)
            qmin = row["Qminimo"]
            qtrans = row["Qtransicion"]

            # ------------------------
            # Validación
            # ------------------------
            if modelo not in interval_models or modelo not in clasif_models:
                resultados.append({
                    **row.to_dict(),
                    "Estado_pred": "SIN MODELO"
                })
                continue

            # ------------------------
            # 1️⃣ Intervalos EQ3
            # ------------------------
            X_pred = pd.DataFrame([[edad, volumen, consumo]],
                                  columns=["Edad", "Volumen", "Consumo_anual"])

            eq_low = interval_models[modelo]["lower"].predict(X_pred)[0]
            eq_up = interval_models[modelo]["upper"].predict(X_pred)[0]
            eq_mid = (eq_low + eq_up)/2

            # ------------------------
            # 2️⃣ Clasificación (sin variable Estado)
            # ------------------------
            X_class = pd.DataFrame([[edad, volumen, qmin, qtrans]],
                                   columns=["Edad", "Volumen", "Qminimo", "Qtransicion"])

            prob = clasif_models[modelo].predict_proba(X_class)[0][1]
            pred = "CUMPLE" if prob >= 0.5 else "NO CUMPLE"

            # ------------------------
            # 3️⃣ Vida útil remanente
            # ------------------------
            degr = float(df_degrad[df_degrad["Descripción"] == modelo]["Degradación_media (%)"].iloc[0])
            if abs(eq_mid) >= EMP_Q3:
                vida = 0
            else:
                t1 = (EMP_Q3 - eq_mid)/degr
                t2 = (-EMP_Q3 - eq_mid)/degr
                validos = [t for t in [t1, t2] if t >= 0]
                vida = min(validos) if validos else MAX_YEARS

            resultados.append({
                **row.to_dict(),
                "EQ3_low": round(eq_low, 3),
                "EQ3_up": round(eq_up, 3),
                "Prob_Cumple": round(prob, 3),
                "Pred_Cumple": pred,
                "Vida_rem": round(vida, 2)
            })

        df_res = pd.DataFrame(resultados)
        st.subheader("📊 Resultados de predicción")
        st.dataframe(df_res)

        st.download_button(
            "📥 Descargar resultados",
            df_res.to_csv(index=False).encode("utf-8"),
            "predicciones.csv",
            "text/csv"
        )
