###############################################################
#              STREAMLIT — PREDICCIÓN MEDIDORES DE AGUA
###############################################################

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import string

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

# Inicialización session_state
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

    if os.path.isdir(MODELOS_INTERVALOS):
        for file in os.listdir(MODELOS_INTERVALOS):
            if file.endswith("_lower.cbm") or file.endswith("_upper.cbm"):
                modelo = file.replace("_lower.cbm","").replace("_upper.cbm","")
                m = CatBoostRegressor()
                m.load_model(os.path.join(MODELOS_INTERVALOS,file))
                interval_models.setdefault(modelo,{})
                if file.endswith("_lower.cbm"):
                    interval_models[modelo]["lower"] = m
                else:
                    interval_models[modelo]["upper"] = m

    if os.path.isdir(MODELOS_CLASIFICACION):
        for file in os.listdir(MODELOS_CLASIFICACION):
            if file.endswith(".cbm"):
                modelo = file.replace(".cbm","")
                m = CatBoostClassifier()
                m.load_model(os.path.join(MODELOS_CLASIFICACION,file))
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
# EXPORTACIÓN A GOOGLE SHEETS (OPTIMIZADA 5 REQUESTS)
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
            cols=str(len(df.columns) + 5)
        )

        data = [df.columns.tolist()] + df.astype(str).values.tolist()
        worksheet.update(data)

        # =======================================
        # FORMATO ANTI-429 — SOLO 5 REQUESTS
        # =======================================

        verde = CellFormat(backgroundColor=Color(0.80, 1.00, 0.80))
        rojo = CellFormat(backgroundColor=Color(1.00, 0.80, 0.80))
        amarillo = CellFormat(backgroundColor=Color(1.00, 1.00, 0.60))
        gris = CellFormat(backgroundColor=Color(0.90, 0.90, 0.90))

        col_pred = df.columns.get_loc("Pred_Conformidad") + 1
        col_riesgo = df.columns.get_loc("Nivel_de_Riesgo") + 1
        col_intervalo = df.columns.get_loc("Dictamen_por_intervalo") + 1

        n_rows = len(df) + 1
        letras = list(string.ascii_uppercase)

        col_pred_l = letras[col_pred - 1]
        col_riesgo_l = letras[col_riesgo - 1]
        col_intervalo_l = letras[col_intervalo - 1]

        # Rangos completos
        pred_range = f"{col_pred_l}2:{col_pred_l}{n_rows}"
        riesgo_range = f"{col_riesgo_l}2:{col_riesgo_l}{n_rows}"
        intervalo_range = f"{col_intervalo_l}2:{col_intervalo_l}{n_rows}"

        # === 1) Formatos base por columna
        format_cell_range(worksheet, pred_range, rojo)
        format_cell_range(worksheet, riesgo_range, rojo)
        format_cell_range(worksheet, intervalo_range, rojo)

        # === 2) Corrección de valores específicos (3 requests más)
        cumple_rows = df.index[df["Pred_Conformidad"] == "CUMPLE"] + 2
        riesgo_bajo = df.index[df["Nivel_de_Riesgo"] == "BAJO"] + 2
        riesgo_medio = df.index[df["Nivel_de_Riesgo"] == "MEDIO"] + 2
        dentro_rows = df.index[df["Dictamen_por_intervalo"] == "DENTRO"] + 2

        for fila in cumple_rows:
            format_cell_range(worksheet, f"{col_pred_l}{fila}", verde)

        for fila in riesgo_bajo:
            format_cell_range(worksheet, f"{col_riesgo_l}{fila}", verde)

        for fila in riesgo_medio:
            format_cell_range(worksheet, f"{col_riesgo_l}{fila}", amarillo)

        for fila in dentro_rows:
            format_cell_range(worksheet, f"{col_intervalo_l}{fila}", verde)

        # Total de requests ≈ 5–8 (MUY eficiente)
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

    st.write("### 🔎 Vista previa de los datos")
    st.dataframe(df_base.head())

    # ===== EJECUTAR PREDICCIÓN =====
    if st.button("🚀 Ejecutar Predicción"):

        df_pred = predecir_lote(df_base, interval_models, clasif_models, df_ic)

        if df_pred.empty:
            st.error("No se pudieron generar predicciones.")
            return

        st.session_state["df_pred"] = df_pred
        st.session_state["prediccion_realizada"] = True
        st.success("Predicción completada correctamente.")

    # ===== MOSTRAR RESULTADOS =====
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
            except Exception:
                st.dataframe(df_pred, use_container_width=True)

            st.download_button(
                "📥 Descargar CSV",
                df_pred.to_csv(index=False),
                "predicciones_medidores.csv"
            )

            st.markdown("### 📤 Exportar a Google Sheets")

            if st.button("Enviar a Google Sheets"):
                ok, msg = exportar_google_sheets(df_pred)
                if ok:
                    st.success(f"Predicción exportada correctamente en: {msg}")
                else:
                    st.error(f"Error al exportar: {msg}")

        # ================= DASHBOARD =================
        with tab2:
            st.write("### 🧮 Indicadores globales")

            col1, col2, col3 = st.columns(3)

            dentro_emp = (df_pred["Dictamen_por_intervalo"] == "DENTRO").mean() * 100
            cumple = (df_pred["Pred_Conformidad"] == "CUMPLE").mean() * 100

            vida_prom = pd.to_numeric(df_pred["Vida_remanente"], errors="coerce").mean()

            col1.metric("Dentro del EMP", f"{dentro_emp:.1f}%")
            col2.metric("CUMPLEN", f"{cumple:.1f}%")
            col3.metric("Vida útil promedio", f"{vida_prom:.1f} años")

            st.markdown("---")

            st.write("### Distribución de riesgo")
            st.bar_chart(
                df_pred["Nivel_de_Riesgo"].value_counts()
                .reindex(["BAJO", "MEDIO", "ALTO", "SIN MODELO"]).fillna(0)
            )

            st.write("### CUMPLE vs NO CUMPLE")
            st.bar_chart(df_pred["Pred_Conformidad"].value_counts())

            st.write("### Vida remanente por modelo")
            vida_mod = (
                df_pred.assign(Vida=pd.to_numeric(df_pred["Vida_remanente"], errors="coerce"))
                .groupby("Modelo_corto")["Vida"].mean().sort_values()
            )
            st.bar_chart(vida_mod)

            st.write("### Histograma EQ3_mid (%)")
            eq = pd.to_numeric(df_pred["EQ3_mid (%)"], errors="coerce").dropna()
            if not eq.empty:
                hist = np.histogram(eq, bins=20)
                st.bar_chart(pd.DataFrame({"bin": hist[1][:-1], "freq": hist[0]}).set_index("bin"))
            else:
                st.info("Sin datos válidos para EQ3_mid (%)")


if __name__ == "__main__":
    main()
