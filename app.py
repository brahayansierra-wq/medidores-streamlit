###############################################################
#              STREAMLIT — PREDICCIÓN MEDIDORES DE AGUA
###############################################################

import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor, CatBoostClassifier

# ============================================================
# CONFIG GENERAL
# ============================================================

st.set_page_config(page_title="Predicción Medidores", layout="wide")

EMP_Q3 = 4
MAX_YEARS = 15

MODELOS_INTERVALOS = "modelos_intervalos"
MODELOS_CLASIFICACION = "modelos_clasificacion"

DEGRADACIONES_FILE = "vida_util_degradacion.csv"

# Google Sheets URL desde secrets
SHEET_URL = st.secrets["sheets"]["url"]

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def safe_float(x):
    try:
        if isinstance(x, list):
            x = x[0]

        x = str(x).replace(",", "").replace("%", "").strip()

        if x in ["", "None", "nan", "NaN", "-", "--"]:
            return 0.0

        return float(x)
    except:
        return 0.0


MAPEO_DESCRIPCION = {
    "ELSTER_V100_Vol_R160_15mm_0.01 (H)": "GB1_R160",
    "ELSTER_GB1_VOL_R160_Q3:2,5_DN15_0,01L": "GB1_R160",
    "ELSTER-GB1-R160-0.01 (H)": "GB1_R160",

    "SAPPEL_ALTAIR_VOL_R160_Q3:2,5_DN15_0,02L": "ALTAIR_R160",
    "DIEHL_ALTAIR_VOL_R160_Q3:2,5_DN15_0,02L": "ALTAIR_R160",

    "SAPPEL_AQUARIUS_Vel_R80_15_0.02": "AQUARIUS_R80",
    "SAPPEL_AQUARIUS_VCU_R80_Q3:2,5_DN15_0,02L": "AQUARIUS_R80",
    "DIEHL_AQUARIUS_VCU_R80_Q3:2,5_DN15_0,02L": "AQUARIUS_R80",

    "HONEYWELL_V200_VOL_R160_Q3:2,5_DN15_0,02L": "V200_R160",
    "ELSTER_V200P_VOL_R160_Q3:2,5_DN15_0,02L": "V200_R160",

    "SENSUS_660_VOL_R315_Q3:2,5_DN15_0,02L": "620_R315",
    "SENSUS_620_VOL_R315_Q3:2,5_DN15_0,02L": "620_R315",
    "SENSUS_620_VOL_R315_Q3:2,5_DN15_0,05L": "620_R315",

    "HONEYWELL_V200_VOL_R315_Q3:2,5_DN15_0,02L": "V200_R315",
    "ELSTER_V200_VOL_R315_Q3:2,5_DN15_0,02L": "V200_R315",

    "ITRON_AQUADIS+_VOL_R315_Q3:2,5_DN15_0,02L": "AQUADIS+_R160",
    "ITRON_AQUADIS+_VOL_R160_Q3:2,5_DN15_0,02L": "AQUADIS+_R160",

    "DIEHL_ALTAIR_VOL_R200_Q3:2,5_DN15_0,02L": "ALTAIR_R315",
    "DIEHL_ALTAIR_VOL_R315_Q3:2,5_DN15_0,02L": "ALTAIR_R315",

    "GEORGE KENT_GKMV40P_VOL_R315_Q3:2,5_DN15_0,02L": "GKMV40P_R315",
}

def obtener_modelo_corto(desc):
    return MAPEO_DESCRIPCION.get(str(desc).strip(), "SIN MODELO")


# ============================================================
# CARGA DE MODELOS
# ============================================================

@st.cache_resource
def cargar_modelos():
    interval_models = {}
    clasif_models = {}

    # MODELOS DE INTERVALOS
    if os.path.isdir(MODELOS_INTERVALOS):
        for file in os.listdir(MODELOS_INTERVALOS):
            if not file.endswith(".cbm"):
                continue

            path = os.path.join(MODELOS_INTERVALOS, file)

            if file.endswith("_lower.cbm"):
                modelo = file.replace("_lower.cbm", "")
                m = CatBoostRegressor()
                m.load_model(path)
                interval_models.setdefault(modelo, {})
                interval_models[modelo]["lower"] = m

            elif file.endswith("_upper.cbm"):
                modelo = file.replace("_upper.cbm", "")
                m = CatBoostRegressor()
                m.load_model(path)
                interval_models.setdefault(modelo, {})
                interval_models[modelo]["upper"] = m

    # MODELOS DE CLASIFICACIÓN
    if os.path.isdir(MODELOS_CLASIFICACION):
        for file in os.listdir(MODELOS_CLASIFICACION):
            if not file.endswith(".cbm"):
                continue

            path = os.path.join(MODELOS_CLASIFICACION, file)
            modelo = file.replace(".cbm", "")
            m = CatBoostClassifier()
            m.load_model(path)
            clasif_models[modelo] = m

    return interval_models, clasif_models


# ============================================================
# FUNCIÓN DE PREDICCIÓN POR LOTE (CORREGIDA)
# ============================================================

def predecir_lote(df, interval_models, clasif_models, df_ic,
                  emp=EMP_Q3, max_years=MAX_YEARS):

    resultados = []

    columnas_necesarias = ["Descripción", "Edad", "Volumen", "Consumo_anual", "Qminimo", "Qtransicion", "Estado"]
    faltantes = [c for c in columnas_necesarias if c not in df.columns]

    if faltantes:
        st.error(f"Faltan columnas requeridas: {faltantes}")
        return pd.DataFrame()

    for _, row in df.iterrows():

        modelo_corto = obtener_modelo_corto(row["Descripción"])

        edad = safe_float(row["Edad"])
        volumen = safe_float(row["Volumen"])
        consumo = safe_float(row["Consumo_anual"])
        qmin = safe_float(row["Qminimo"])
        qtrans = safe_float(row["Qtransicion"])
        estado = str(row["Estado"])

        # ============================
        # 1. INTERVALOS
        # ============================

        eq3_low = eq3_high = eq3_mid = None

        if modelo_corto in interval_models:
            m_interval = interval_models[modelo_corto]

            if "lower" in m_interval and "upper" in m_interval:
                X_pred = pd.DataFrame([[edad, volumen, consumo]],
                                      columns=["Edad", "Volumen", "Consumo_anual"])
                eq3_low = float(m_interval["lower"].predict(X_pred)[0])
                eq3_high = float(m_interval["upper"].predict(X_pred)[0])
                eq3_mid = (eq3_low + eq3_high) / 2.0

        # ============================
        # 2. CLASIFICACIÓN (CORREGIDO)
        # ============================

        prob = None
        pred = "SIN MODELO"

        if modelo_corto in clasif_models:

            features_modelo = clasif_models[modelo_corto].feature_names_

            # Construir diccionario con TODAS las columnas posibles
            row_dict = {
                "Edad": edad,
                "Volumen": volumen,
                "Qminimo": qmin,
                "Qtransicion": qtrans,
                "Estado": estado,        # la columna faltante que causaba error
                "Consumo_anual": consumo # por si el modelo la necesita
            }

            # Construir DataFrame siguiendo EL ORDEN EXACTO del modelo
            X_class = pd.DataFrame(
                [[row_dict[f] for f in features_modelo]],
                columns=features_modelo
            )

            p = clasif_models[modelo_corto].predict_proba(X_class)[0, 1]
            prob = float(p)
            pred = "CUMPLE" if prob >= 0.5 else "NO CUMPLE"

        # ============================
        # 3. VIDA ÚTIL
        # ============================

        vida_rem = None
        dictamen = "SIN MODELO"

        if (eq3_mid is not None) and \
           ("Descripción" in df_ic.columns) and \
           ("Degradación_media (%)" in df_ic.columns):

            if modelo_corto in df_ic["Descripción"].values:
                degr = float(df_ic.loc[df_ic["Descripción"] == modelo_corto,
                                       "Degradación_media (%)"].values[0])

                if degr != 0:
                    t1 = (emp - eq3_mid) / degr
                    t2 = (-emp - eq3_mid) / degr
                    valid = [t for t in [t1, t2] if t >= 0]
                    vida_rem = min(valid) if valid else max_years

        if eq3_mid is not None:
            dictamen = "FUERA" if abs(eq3_mid) > emp else "DENTRO"

        resultados.append({
            **row.to_dict(),
            "Modelo_corto": modelo_corto,
            "EQ3_lower (%)": eq3_low,
            "EQ3_upper (%)": eq3_high,
            "EQ3_mid (%)": eq3_mid,
            "Probabilidad_Cumple": prob,
            "Pred_Conformidad": pred,
            "Vida_remanente": vida_rem,
            "Dictamen_por_intervalo": dictamen
        })

    return pd.DataFrame(resultados)


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================

def main():

    st.title("🔮 Sistema Predictivo de Medidores de Agua")
    st.markdown("---")

    opcion = st.sidebar.radio("Fuente de datos:", ["Google Sheets", "Subir archivo CSV"])

    interval_models, clasif_models = cargar_modelos()

    df_ic = pd.read_csv(DEGRADACIONES_FILE)

    if opcion == "Google Sheets":
        try:
            if SHEET_URL.endswith("output=xlsx"):
                df_base = pd.read_excel(SHEET_URL)
            else:
                df_base = pd.read_csv(SHEET_URL)

            st.success("Datos cargados desde Google Sheets.")

        except Exception as e:
            st.error(f"Error cargando Google Sheets: {e}")
            return

    else:
        file = st.file_uploader("Sube un archivo CSV", type=["csv"])

        if not file:
            st.info("Sube un archivo CSV para continuar.")
            return

        df_base = pd.read_csv(file)
        st.success("Archivo CSV cargado correctamente.")

    st.write("### 🔎 Vista previa")
    st.dataframe(df_base.head())

    if st.button("🚀 Ejecutar Predicción"):
        df_pred = predecir_lote(df_base, interval_models, clasif_models, df_ic)

        if df_pred.empty:
            st.error("No se pudieron generar las predicciones.")
            return

        st.success("Predicción completada.")
        st.dataframe(df_pred)

        st.download_button(
            "📥 Descargar CSV",
            df_pred.to_csv(index=False),
            "predicciones_medidores.csv",
            "text/csv"
        )


if __name__ == "__main__":
    main()
