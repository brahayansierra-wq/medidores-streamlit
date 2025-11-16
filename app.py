###############################################################
#                     STREAMLIT — PREDICCIÓN MEDIDORES
###############################################################

import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor, CatBoostClassifier

# ============================================================
# CONFIGURACIÓN
# ============================================================

EMP_Q3 = 4 - (4/3)   # ≈ 2.67%
MAX_YEARS = 15      
SHEET_URL = st.secrets["google_sheets_url"]   # ⚠ ENSÉÑAME ESTE VALOR
MODELS_PATH = "modelos"                       # Carpeta con .cbm
DEGRADACIONES_FILE = "vida_util_degradacion.csv"

st.set_page_config(page_title="Predicción Medidores", layout="wide")

# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(x):
    try:
        if isinstance(x, list):
            x = x[0]

        x = str(x).replace(",", "").replace("%", "").strip()

        if x in ["", "None", "nan", "-", "--"]:
            return 0.0

        return float(x)
    except:
        return 0.0


# ============================================================
# MAPEO DE DESCRIPCIÓN LARGA → MODELO CORTO
# ============================================================

MAPEO_DESCRIPCION = {
    # GB1_R160
    "ELSTER_V100_Vol_R160_15mm_0.01 (H)": "GB1_R160",
    "ELSTER_GB1_VOL_R160_Q3:2,5_DN15_0,01L": "GB1_R160",
    "ELSTER-GB1-R160-0.01 (H)": "GB1_R160",

    # ALTAIR R160
    "SAPPEL_ALTAIR_VOL_R160_Q3:2,5_DN15_0,02L": "ALTAIR_R160",
    "DIEHL_ALTAIR_VOL_R160_Q3:2,5_DN15_0,02L": "ALTAIR_R160",

    # AQUARIUS R80
    "SAPPEL_AQUARIUS_Vel_R80_15_0.02": "AQUARIUS_R80",
    "SAPPEL_AQUARIUS_VCU_R80_Q3:2,5_DN15_0,02L": "AQUARIUS_R80",
    "DIEHL_AQUARIUS_VCU_R80_Q3:2,5_DN15_0,02L": "AQUARIUS_R80",

    # V200 R160
    "HONEYWELL_V200_VOL_R160_Q3:2,5_DN15_0,02L": "V200_R160",
    "ELSTER_V200P_VOL_R160_Q3:2,5_DN15_0,02L": "V200_R160",

    # 620_R315
    "SENSUS_660_VOL_R315_Q3:2,5_DN15_0,02L": "620_R315",
    "SENSUS_620_VOL_R315_Q3:2,5_DN15_0,02L": "620_R315",
    "SENSUS_620_VOL_R315_Q3:2,5_DN15_0,05L": "620_R315",

    # V200 R315
    "HONEYWELL_V200_VOL_R315_Q3:2,5_DN15_0,02L": "V200_R315",
    "ELSTER_V200_VOL_R315_Q3:2,5_DN15_0,02L": "V200_R315",

    # AQUADIS+ R160
    "ITRON_AQUADIS+_VOL_R315_Q3:2,5_DN15_0,02L": "AQUADIS+_R160",
    "ITRON_AQUADIS+_VOL_R160_Q3:2,5_DN15_0,02L": "AQUADIS+_R160",

    # ALTAIR R315
    "DIEHL_ALTAIR_VOL_R200_Q3:2,5_DN15_0,02L": "ALTAIR_R315",
    "DIEHL_ALTAIR_VOL_R315_Q3:2,5_DN15_0,02L": "ALTAIR_R315",

    # GKMV40P_R315
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

    for file in os.listdir(MODELS_PATH):
        path = os.path.join(MODELS_PATH, file)

        if file.endswith("_lower.cbm"):
            modelo = file.replace("_lower.cbm", "")
            interval_models.setdefault(modelo, {})
            interval_models[modelo]["lower"] = CatBoostRegressor().load_model(path)

        elif file.endswith("_upper.cbm"):
            modelo = file.replace("_upper.cbm", "")
            interval_models.setdefault(modelo, {})
            interval_models[modelo]["upper"] = CatBoostRegressor().load_model(path)

        elif file.endswith(".cbm"):
            modelo = file.replace(".cbm", "")
            clasif_models[modelo] = CatBoostClassifier().load_model(path)

    return interval_models, clasif_models


# ============================================================
# PREDICCIÓN
# ============================================================

def predecir_lote(df, interval_models, clasif_models, df_ic, emp=EMP_Q3, max_years=MAX_YEARS):

    resultados = []

    for _, row in df.iterrows():
        modelo_corto = obtener_modelo_corto(row["Descripción"])

        edad = safe_float(row.get("Edad"))
        volumen = safe_float(row.get("Volumen"))
        consumo = safe_float(row.get("Consumo_anual"))
        qmin = safe_float(row.get("Qminimo"))
        qtrans = safe_float(row.get("Qtransicion"))

        # 1. Predicción intervalos
        if modelo_corto in interval_models:
            X_pred = pd.DataFrame([[edad, volumen, consumo]],
                                  columns=["Edad", "Volumen", "Consumo_anual"])
            eq3_low = interval_models[modelo_corto]["lower"].predict(X_pred)[0]
            eq3_high = interval_models[modelo_corto]["upper"].predict(X_pred)[0]
            eq3_mid = (eq3_low + eq3_high) / 2
        else:
            eq3_low = eq3_high = eq3_mid = None

        # 2. Clasificación
        if modelo_corto in clasif_models:
            X_class = pd.DataFrame([[edad, volumen, qmin, qtrans]],
                                   columns=["Edad", "Volumen", "Qminimo", "Qtransicion"])
            prob = clasif_models[modelo_corto].predict_proba(X_class)[0,1]
            pred = "CUMPLE" if prob >= 0.5 else "NO CUMPLE"
        else:
            prob = None
            pred = "SIN MODELO"

        # 3. Vida útil
        vida_rem = None
        if eq3_mid is not None and modelo_corto in df_ic["Descripción"].values:
            degr = float(df_ic.loc[df_ic["Descripción"] == modelo_corto, "Degradación_media (%)"].values[0])
            if degr != 0:
                t1 = (emp - eq3_mid) / degr
                t2 = (-emp - eq3_mid) / degr
                valid = [t for t in [t1, t2] if t >= 0]
                vida_rem = min(valid) if valid else max_years

        # Dictamen
        if eq3_mid is None:
            dictamen = "SIN MODELO"
        else:
            dictamen = "FUERA" if abs(eq3_mid) > emp else "DENTRO"

        resultados.append({
            **row.to_dict(),
            "Modelo_corto": modelo_corto,
            "EQ3_lower (%)": eq3_low,
            "EQ3_upper (%)": eq3_high,
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
    st.title("🔮 Sistema Predictivo de Medidores de Agua — CatBoost + Intervalos")

    st.sidebar.header("⚙ Configuración")

    # 1. Cargar modelos
    interval_models, clasif_models = cargar_modelos()

    # 2. Cargar degradaciones
    df_ic = pd.read_csv(DEGRADACIONES_FILE)

    # 3. Selección de fuente de datos
    opcion = st.sidebar.radio("Fuente de datos:", ["Google Sheets", "Subir archivo CSV"])

    if opcion == "Google Sheets":
        try:
            df_base = pd.read_csv(SHEET_URL)
            st.success("Google Sheets cargado correctamente.")
        except:
            st.error("Error cargando Google Sheets.")
            return

    else:
        file = st.file_uploader("Sube un archivo CSV")
        if file:
            df_base = pd.read_csv(file)
        else:
            st.info("Sube un archivo para continuar.")
            return

    st.write("### 📄 Vista previa de datos")
    st.dataframe(df_base.head())

    if st.button("🚀 Ejecutar Predicción"):
        df_pred = predecir_lote(df_base, interval_models, clasif_models, df_ic)
        st.success("Predicción completada.")

        st.write("### 📊 Resultados")
        st.dataframe(df_pred)

        st.download_button(
            label="📥 Descargar resultados",
            data=df_pred.to_csv(index=False),
            file_name="predicciones_medidores.csv"
        )


if __name__ == "__main__":
    main()

