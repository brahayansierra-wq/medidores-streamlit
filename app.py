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

EMP_Q3 = 3.2            # ≈ 2.67% (umbral metrológico)
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
# FUNCIÓN DE PREDICCIÓN POR LOTE
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

        # ======================================
        # 1. INTERVALOS EQ3
        # ======================================

        eq3_low = eq3_high = eq3_mid = None

        if modelo_corto in interval_models:
            m_int = interval_models[modelo_corto]

            if "lower" in m_int and "upper" in m_int:
                X_pred = pd.DataFrame([[edad, volumen, consumo]],
                                      columns=["Edad", "Volumen", "Consumo_anual"])
                eq3_low = float(m_int["lower"].predict(X_pred)[0])
                eq3_high = float(m_int["upper"].predict(X_pred)[0])
                eq3_mid = (eq3_low + eq3_high) / 2.0

        # ======================================
        # 2. CLASIFICACIÓN CatBoost
        # ======================================

        prob = None
        pred = "SIN MODELO"
        riesgo = "SIN MODELO"

        if modelo_corto in clasif_models:

            features_modelo = clasif_models[modelo_corto].feature_names_

            row_dict = {
                "Edad": edad,
                "Volumen": volumen,
                "Qminimo": qmin,
                "Qtransicion": qtrans,
                "Estado": estado,
                "Consumo_anual": consumo,
            }

            X_class = pd.DataFrame(
                [[row_dict[f] for f in features_modelo]],
                columns=features_modelo
            )

            prob = float(clasif_models[modelo_corto].predict_proba(X_class)[0, 1])
            pred = "CUMPLE" if prob >= 0.5 else "NO CUMPLE"

            # ===== NIVEL DE RIESGO (SEMÁFORO) =====
            if prob >= 0.80:
                riesgo = "BAJO"
            elif prob >= 0.50:
                riesgo = "MEDIO"
            else:
                riesgo = "ALTO"

        # ======================================
        # 3. VIDA ÚTIL REMANENTE
        # ======================================

        vida_rem = None

        if "Descripción" in df_ic.columns and modelo_corto in df_ic["Descripción"].values:

            degr = float(df_ic.loc[
                df_ic["Descripción"] == modelo_corto,
                "Degradación_media (%)"
            ].values[0])

            eq0 = eq3_mid

            if eq0 is None or abs(eq0) >= emp or degr == 0:
                vida_rem = 0
            else:
                t1 = (emp - eq0) / degr
                t2 = (-emp - eq0) / degr
                valid = [t for t in (t1, t2) if t >= 0]
                vida_rem = min(valid) if valid else max_years

        # ======================================
        # 4. DICTAMEN
        # ======================================

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
# ESTILO: SEMAFORIZACIÓN EN TABLA
# ============================================================

def color_semaforo(val):
    if val == "DENTRO":
        return "background-color: #d4edda;"  # verde suave
    if val == "FUERA":
        return "background-color: #f8d7da;"  # rojo suave
    if val == "CUMPLE":
        return "background-color: #cce5ff;"  # azul claro
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
# INTERFAZ STREAMLIT
# ============================================================

def main():

    st.title("🔮 Sistema Predictivo de Medidores de Agua")
    st.markdown("---")

    opcion = st.sidebar.radio("Fuente de datos:", ["Google Sheets", "Subir archivo CSV"])

    interval_models, clasif_models = cargar_modelos()

    df_ic = pd.read_csv(DEGRADACIONES_FILE)

    # ----------------- CARGA DE DATOS -----------------
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

    # ----------------- PREDICCIÓN -----------------
    if st.button("🚀 Ejecutar Predicción"):
        df_pred = predecir_lote(df_base, interval_models, clasif_models, df_ic)

        if df_pred.empty:
            st.error("No se pudieron generar las predicciones.")
            return

        st.success("Predicción completada.")

        # Pestañas: Resultados y Dashboard
        tab1, tab2 = st.tabs(["📄 Resultados", "📊 Dashboard"])

        # ==================== TABLA + SEMÁFORO ====================
        with tab1:
            st.write("### 📊 Resultados detallados")

            # Tabla con estilos
            try:
                styled = df_pred.style.applymap(
                    color_semaforo,
                    subset=["Pred_Conformidad", "Dictamen_por_intervalo", "Nivel_de_Riesgo"]
                )
                st.dataframe(styled, use_container_width=True)
            except Exception:
                # Si hay algún problema con Styler, mostramos la tabla normal
                st.dataframe(df_pred, use_container_width=True)

            st.download_button(
                "📥 Descargar CSV",
                df_pred.to_csv(index=False),
                "predicciones_medidores.csv",
                "text/csv"
            )

        # ==================== DASHBOARD ====================
        with tab2:
            st.write("### 🧮 Indicadores globales")

            col1, col2, col3 = st.columns(3)

            # Medidores dentro del EMP
            dentro_emp = (df_pred["Dictamen_por_intervalo"] == "DENTRO").mean()
            dentro_emp = dentro_emp * 100 if not np.isnan(dentro_emp) else 0

            # Medidores que cumplen
            cumple = (df_pred["Pred_Conformidad"] == "CUMPLE").mean()
            cumple = cumple * 100 if not np.isnan(cumple) else 0

            # Vida remanente promedio
            vida_series = pd.to_numeric(df_pred["Vida_remanente"], errors="coerce")
            vida_prom = vida_series.mean() if not vida_series.empty else np.nan

            with col1:
                st.metric("Medidores dentro del EMP", f"{dentro_emp:.1f}%")
            with col2:
                st.metric("Medidores que CUMPLEN", f"{cumple:.1f}%")
            with col3:
                if np.isnan(vida_prom):
                    st.metric("Vida remanente promedio", "N/D")
                else:
                    st.metric("Vida remanente promedio", f"{vida_prom:.1f} años")

            st.markdown("---")
            st.write("### Distribución de riesgo")

            if "Nivel_de_Riesgo" in df_pred.columns:
                riesgo_counts = df_pred["Nivel_de_Riesgo"].value_counts().reindex(
                    ["BAJO", "MEDIO", "ALTO", "SIN MODELO"]
                ).fillna(0)
                st.bar_chart(riesgo_counts)
            else:
                st.info("No se encontró la columna 'Nivel_de_Riesgo'.")

            st.markdown("### Distribución CUMPLE / NO CUMPLE")
            if "Pred_Conformidad" in df_pred.columns:
                cumple_counts = df_pred["Pred_Conformidad"].value_counts()
                st.bar_chart(cumple_counts)
            else:
                st.info("No se encontró la columna 'Pred_Conformidad'.")

            st.markdown("### Vida remanente promedio por modelo")
            if "Modelo_corto" in df_pred.columns and "Vida_remanente" in df_pred.columns:
                df_vida_modelo = (
                    df_pred
                    .assign(Vida_remanente_num=pd.to_numeric(df_pred["Vida_remanente"], errors="coerce"))
                    .groupby("Modelo_corto")["Vida_remanente_num"]
                    .mean()
                    .sort_values()
                )
                st.bar_chart(df_vida_modelo)
            else:
                st.info("No se puede calcular la vida remanente por modelo.")

            st.markdown("### Histograma de EQ3_mid (%)")
            if "EQ3_mid (%)" in df_pred.columns:
                eq = pd.to_numeric(df_pred["EQ3_mid (%)"], errors="coerce").dropna()
                if not eq.empty:
                    # Usamos value_counts por bins para evitar importar matplotlib
                    hist = np.histogram(eq, bins=20)
                    hist_df = pd.DataFrame({"bin": hist[1][:-1].astype(str), "freq": hist[0]})
                    st.bar_chart(hist_df.set_index("bin"))
                else:
                    st.info("No hay datos válidos de EQ3_mid (%).")
            else:
                st.info("No se encontró la columna 'EQ3_mid (%)'.")


if __name__ == "__main__":
    main()

