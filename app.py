###############################################################
#              STREAMLIT — PREDICCIÓN MEDIDORES DE AGUA
###############################################################

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

from catboost import CatBoostRegressor, CatBoostClassifier

import gspread
from google.oauth2.service_account import Credentials
from gspread_formatting import (
    format_cell_range,
    CellFormat,
    Color
)

# ============================================================
# CONFIG GENERAL
# ============================================================

st.set_page_config(page_title="Predicción Medidores", layout="wide")

EMP_Q3 = 4 - (4/3)  # ≈ 2.67%
MAX_YEARS = 15

MODELOS_INTERVALOS = "modelos_intervalos"
MODELOS_CLASIFICACION = "modelos_clasificacion"

DEGRADACIONES_FILE = "vida_util_degradacion.csv"

SHEET_URL = st.secrets["sheets"]["url"]

# Inicialización de session_state
if "prediccion_realizada" not in st.session_state:
    st.session_state["prediccion_realizada"] = False
if "df_pred" not in st.session_state:
    st.session_state["df_pred"] = None


# ============================================================
# FUNCIONES AUXILIARES
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

    # Intervalos
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

    # Clasificación
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
# FUNCIÓN DE PREDICCIÓN
# ============================================================

def predecir_lote(df, interval_models, clasif_models, df_ic,
                  emp=EMP_Q3, max_years=MAX_YEARS):

    resultados = []

    for _, row in df.iterrows():

        modelo_corto = obtener_modelo_corto(row["Descripción"])

        edad = safe_float(row["Edad"])
        volumen = safe_float(row["Volumen"])
        consumo = safe_float(row["Consumo_anual"])
        qmin = safe_float(row["Qminimo"])
        qtrans = safe_float(row["Qtransicion"])
        estado = str(row["Estado"])

        # ===== INTERVALOS =====
        eq3_low = eq3_high = eq3_mid = None

        if modelo_corto in interval_models:
            m = interval_models[modelo_corto]
            if "lower" in m and "upper" in m:
                Xp = pd.DataFrame([[edad, volumen, consumo]],
                                  columns=["Edad", "Volumen", "Consumo_anual"])
                eq3_low = float(m["lower"].predict(Xp)[0])
                eq3_high = float(m["upper"].predict(Xp)[0])
                eq3_mid = (eq3_low + eq3_high) / 2

        # ===== CLASIFICACIÓN =====
        prob = None
        pred = "SIN MODELO"
        riesgo = "SIN MODELO"

        if modelo_corto in clasif_models:
            features = clasif_models[modelo_corto].feature_names_

            fila = {
                "Edad": edad,
                "Volumen": volumen,
                "Consumo_anual": consumo,
                "Qminimo": qmin,
                "Qtransicion": qtrans,
                "Estado": estado
            }

            X_class = pd.DataFrame([[fila[f] for f in features]], columns=features)
            prob = float(clasif_models[modelo_corto].predict_proba(X_class)[0, 1])

            pred = "CUMPLE" if prob >= 0.5 else "NO CUMPLE"

            if prob >= 0.80:
                riesgo = "BAJO"
            elif prob >= 0.50:
                riesgo = "MEDIO"
            else:
                riesgo = "ALTO"

        # ===== VIDA REMANENTE =====
        vida_rem = None

        if "Descripción" in df_ic.columns and modelo_corto in df_ic["Descripción"].values:
            degr = float(df_ic.loc[df_ic["Descripción"] == modelo_corto,
                                   "Degradación_media (%)"].values[0])

            eq0 = eq3_mid

            if eq0 is None or abs(eq0) >= emp or degr == 0:
                vida_rem = 0
            else:
                t1 = (emp - eq0) / degr
                t2 = (-emp - eq0) / degr
                valid = [t for t in (t1, t2) if t >= 0]
                vida_rem = min(valid) if valid else max_years

        # ===== DICTAMEN =====
        if eq3_mid is None:
            dictamen = "SIN MODELO"
        else:
            dictamen = "FUERA" if abs(eq3_mid) > emp else "DENTRO"

        resultados.append({
            **row.to_dict(),
            "Modelo_corto": modelo_corto,
            "EQ3_lower (%)": eq3_low,
            "EQ3_upper (%)": eq3_high,
            "EQ3_mid (%)": eq3_mid,
            "Probabilidad_Cumple": prob,
            "Pred_Conformidad": pred,
            "Nivel_de_Riesgo": riesgo,
            "Vida_remanente": vida_rem,
            "Dictamen_por_intervalo": dictamen
        })

    return pd.DataFrame(resultados)


# ============================================================
# SEMÁFORO EN TABLA STREAMLIT
# ============================================================

def color_semaforo(val):
    if val == "DENTRO":
        return "background-color: #d4edda;"
    if val == "FUERA":
        return "background-color: #f8d7da;"
    if val == "CUMPLE":
        return "background-color: #cce5ff;"
    if val == "NO CUMPLE":
        return "background-color: #f8d7da;"
    if val == "BAJO":
        return "background-color: #d4edda;"
    if val == "MEDIO":
        return "background-color: #fff3cd;"
    if val == "ALTO":
        return "background-color: #f8d7da;"
    return ""


# ============================================================
# EXPORTACIÓN A GOOGLE SHEETS
# ============================================================

def exportar_google_sheets(df):
    try:
        creds_info = st.secrets["gcp_service_account"]
        sheet_id = st.secrets["google_sheets"]["sheet_id"]

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]

        creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
        client = gspread.authorize(creds)

        sheet = client.open_by_key(sheet_id)

        fecha = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        nombre_hoja = f"Predicciones_{fecha}"

        worksheet = sheet.add_worksheet(
            title=nombre_hoja,
            rows=str(len(df) + 10),
            cols=str(len(df.columns) + 10)
        )

        # Subir datos
        data = [df.columns.tolist()] + df.astype(str).values.tolist()
        worksheet.update(data)

        # -------- GENERADOR DE LETRAS DE COLUMNA --------
        def col_letter(idx):  # idx es índice 1-based
            result = ""
            while idx > 0:
                idx, rem = divmod(idx - 1, 26)
                result = chr(65 + rem) + result
            return result

        # Columnas clave
        col_pred = df.columns.get_loc("Pred_Conformidad") + 1
        col_riesgo = df.columns.get_loc("Nivel_de_Riesgo") + 1
        col_intervalo = df.columns.get_loc("Dictamen_por_intervalo") + 1

        n_rows = len(df) + 1

        verde = CellFormat(backgroundColor=Color(0.80, 1.00, 0.80))
        rojo = CellFormat(backgroundColor=Color(1.00, 0.80, 0.80))
        amarillo = CellFormat(backgroundColor=Color(1.00, 1.00, 0.60))
        gris = CellFormat(backgroundColor=Color(0.90, 0.90, 0.90))

        # --------- FORMATO EN RANGOS GRANDES (EFICIENTE) ---------

        # Pred_Conformidad
        valores_pred = df["Pred_Conformidad"].tolist()
        col_L = col_letter(col_pred)
        for i, v in enumerate(valores_pred, start=2):
            fmt = verde if v == "CUMPLE" else rojo
            format_cell_range(worksheet, f"{col_L}{i}", fmt)

        # Nivel_de_Riesgo
        valores_riesgo = df["Nivel_de_Riesgo"].tolist()
        col_R = col_letter(col_riesgo)
        for i, v in enumerate(valores_riesgo, start=2):
            if v == "BAJO": fmt = verde
            elif v == "MEDIO": fmt = amarillo
            elif v == "ALTO": fmt = rojo
            else: fmt = gris
            format_cell_range(worksheet, f"{col_R}{i}", fmt)

        # Dictamen
        valores_dict = df["Dictamen_por_intervalo"].tolist()
        col_D = col_letter(col_intervalo)
        for i, v in enumerate(valores_dict, start=2):
            fmt = verde if v == "DENTRO" else rojo
            format_cell_range(worksheet, f"{col_D}{i}", fmt)

        return True, nombre_hoja

    except Exception as e:
        return False, str(e)



# ============================================================
# INTERFAZ STREAMLIT
# ============================================================

def main():
    st.title("🔮 Sistema Predictivo de Medidores de Agua")
    st.markdown("---")

    opcion = st.sidebar.radio("Fuente de datos:", ["Google Sheets", "Subir archivo CSV"])

    interval_models, clasif_models = cargar_modelos()
    df_ic = pd.read_csv(DEGRADACIONES_FILE)

    # ===== CARGA DE DATOS =====
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

    # ===== EJECUTAR PREDICCIÓN =====
    if st.button("🚀 Ejecutar Predicción"):

        df_pred = predecir_lote(df_base, interval_models, clasif_models, df_ic)

        if df_pred.empty:
            st.error("No se pudieron generar predicciones.")
            return

        st.session_state["df_pred"] = df_pred
        st.session_state["prediccion_realizada"] = True
        st.success("Predicción completada.")

    # Mostrar resultados solo si ya se generó predicción
    if st.session_state["prediccion_realizada"]:
        df_pred = st.session_state["df_pred"]

        tab1, tab2 = st.tabs(["📄 Resultados", "📊 Dashboard"])

        # ================= TABLA =================
        with tab1:
            st.write("### 📊 Resultados detallados")

            try:
                styled = df_pred.style.applymap(
                    color_semaforo,
                    subset=["Pred_Conformidad", "Dictamen_por_intervalo", "Nivel_de_Riesgo"]
                )
                st.dataframe(styled, use_container_width=True)
            except:
                st.dataframe(df_pred, use_container_width=True)

            st.download_button(
                "📥 Descargar CSV",
                df_pred.to_csv(index=False),
                "predicciones_medidores.csv"
            )

            # ===== EXPORTAR GOOGLE SHEETS =====
            st.markdown("### 📤 Exportar a Google Sheets")

            if st.button("Enviar a Google Sheets"):
                ok, msg = exportar_google_sheets(df_pred)
                if ok:
                    st.success(f"Predicción exportada correctamente en la hoja: {msg}")
                else:
                    st.error(f"Error: {msg}")

        # ================= DASHBOARD =================
        with tab2:
            st.write("### 🧮 Indicadores globales")

            col1, col2, col3 = st.columns(3)

            dentro_emp = (df_pred["Dictamen_por_intervalo"] == "DENTRO").mean() * 100
            cumple = (df_pred["Pred_Conformidad"] == "CUMPLE").mean() * 100

            vida_series = pd.to_numeric(df_pred["Vida_remanente"], errors="coerce")
            vida_prom = vida_series.mean()

            col1.metric("Medidores dentro del EMP", f"{dentro_emp:.1f}%")
            col2.metric("Medidores que CUMPLEN", f"{cumple:.1f}%")
            col3.metric("Vida remanente promedio", f"{vida_prom:.1f} años")

            st.markdown("---")

            st.write("### Distribución de riesgo")
            st.bar_chart(
                df_pred["Nivel_de_Riesgo"].value_counts()
                .reindex(["BAJO", "MEDIO", "ALTO", "SIN MODELO"])
                .fillna(0)
            )

            st.write("### Distribución CUMPLE / NO CUMPLE")
            st.bar_chart(df_pred["Pred_Conformidad"].value_counts())

            st.write("### Vida remanente promedio por modelo")
            vida_mod = (
                df_pred.assign(Vida=pd.to_numeric(df_pred["Vida_remanente"], errors="coerce"))
                .groupby("Modelo_corto")["Vida"].mean().sort_values()
            )
            st.bar_chart(vida_mod)

            st.write("### Histograma de EQ3_mid (%)")
            eq = pd.to_numeric(df_pred["EQ3_mid (%)"], errors="coerce").dropna()
            if not eq.empty:
                hist = np.histogram(eq, bins=20)
                hist_df = pd.DataFrame({"bin": hist[1][:-1].astype(str), "freq": hist[0]})
                st.bar_chart(hist_df.set_index("bin"))
            else:
                st.info("No hay datos válidos para EQ3_mid (%)")


if __name__ == "__main__":
    main()
