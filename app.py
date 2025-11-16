import streamlit as st
import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

# ============================================================
# ⚙️ CONFIGURACIÓN GLOBAL
# ============================================================

# Umbral EMP ajustado para EQ3 (±[4 - 4/3] %)
EMP_Q3 = 4 #- (4 / 3)      # ≈ 2.67 %
MAX_YEARS = 15            # límite máximo de simulación de vida remanente (años)

# ⚠️ Pega aquí la URL CSV publicada de tu Google Sheets
DEFAULT_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTh6UM90hkGvj-aqIByeP7MClXR_kjkt2EmIwF_vuMwkFyvLyuG2YTtotwG0A_GaYa8B7hIC0f4SLox/pub?gid=0&single=true&output=csv"

# ============================================================
# 🗺️ MAPEO DESCRIPCIÓN LARGA → MODELO CORTO
# ============================================================

MAPA_DESCRIPCION_A_MODELO = {
    # GB1_R160
    "ELSTER_V100_Vol_R160_15mm_0.01 (H)": "GB1_R160",
    "ELSTER_GB1_VOL_R160_Q3:2,5_DN15_0,01L": "GB1_R160",
    "ELSTER-GB1-R160-0.01 (H)": "GB1_R160",

    # ALTAIR_R160
    "SAPPEL_ALTAIR_VOL_R160_Q3:2,5_DN15_0,02L": "ALTAIR_R160",
    "DIEHL_ALTAIR_VOL_R160_Q3:2,5_DN15_0,02L": "ALTAIR_R160",

    # AQUARIUS_R80
    "SAPPEL_AQUARIUS_Vel_R80_15_0.02": "AQUARIUS_R80",
    "SAPPEL_AQUARIUS_VCU_R80_Q3:2,5_DN15_0,02L": "AQUARIUS_R80",
    "DIEHL_AQUARIUS_VCU_R80_Q3:2,5_DN15_0,02L": "AQUARIUS_R80",

    # V200_R160
    "HONEYWELL_V200_VOL_R160_Q3:2,5_DN15_0,02L": "V200_R160",
    "ELSTER_V200P_VOL_R160_Q3:2,5_DN15_0,02L": "V200_R160",

    # 620_R315
    "SENSUS_660_VOL_R315_Q3:2,5_DN15_0,02L": "620_R315",
    "SENSUS_620_VOL_R315_Q3:2,5_DN15_0,02L": "620_R315",
    "SENSUS_620_VOL_R315_Q3:2,5_DN15_0,05L": "620_R315",

    # V200_R315
    "HONEYWELL_V200_VOL_R315_Q3:2,5_DN15_0,02L": "V200_R315",
    "ELSTER_V200_VOL_R315_Q3:2,5_DN15_0,02L": "V200_R315",

    # AQUADIS+_R160
    "ITRON_AQUADIS+_VOL_R315_Q3:2,5_DN15_0,02L": "AQUADIS+_R160",
    "ITRON_AQUADIS+_VOL_R160_Q3:2,5_DN15_0,02L": "AQUADIS+_R160",

    # ALTAIR_R315
    "DIEHL_ALTAIR_VOL_R200_Q3:2,5_DN15_0,02L": "ALTAIR_R315",
    "DIEHL_ALTAIR_VOL_R315_Q3:2,5_DN15_0,02L": "ALTAIR_R315",

    # GKMV40P_R315
    "GEORGE KENT_GKMV40P_VOL_R315_Q3:2,5_DN15_0,02L": "GKMV40P_R315",
}

MODELOS_VALIDOS = [
    "620_R315", "ALTAIR_R160", "ALTAIR_R315", "AQUADIS+_R160",
    "AQUARIUS_R80", "GB1_R160", "GKMV40P_R315", "V200_R160", "V200_R315"
]

# ============================================================
# 📥 CARGA DE MODELOS Y TABLA DE DEGRADACIÓN
# ============================================================

@st.cache_resource
def cargar_modelos():
    """
    Carga todos los modelos de intervalos y clasificación
    desde las carpetas locales del repo:
      - modelos_intervalos/
      - modelos_clasificacion/
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    dir_intervalos = os.path.join(base_path, "modelos_intervalos")
    dir_clasificacion = os.path.join(base_path, "modelos_clasificacion")

    interval_models = {}
    clasif_models = {}

    # Modelos de intervalos (lower/upper)
    if os.path.isdir(dir_intervalos):
        for file in os.listdir(dir_intervalos):
            path = os.path.join(dir_intervalos, file)
            if file.endswith("_lower.cbm"):
                name = file.replace("_lower.cbm", "")
                m = CatBoostRegressor()
                m.load_model(path)
                interval_models.setdefault(name, {})["lower"] = m
            elif file.endswith("_upper.cbm"):
                name = file.replace("_upper.cbm", "")
                m = CatBoostRegressor()
                m.load_model(path)
                interval_models.setdefault(name, {})["upper"] = m

    # Modelos de clasificación
    if os.path.isdir(dir_clasificacion):
        for file in os.listdir(dir_clasificacion):
            if file.endswith(".cbm"):
                name = file.replace(".cbm", "")
                path = os.path.join(dir_clasificacion, file)
                m = CatBoostClassifier()
                m.load_model(path)
                clasif_models[name] = m

    return interval_models, clasif_models


@st.cache_data
def cargar_degradaciones():
    base_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_path, "data", "vida_util_degradacion.csv")
    df_ic = pd.read_csv(path)
    return df_ic


# ============================================================
# 📊 LECTURA DE DATOS DESDE GOOGLE SHEETS
# ============================================================

@st.cache_data
def leer_base_medidores(url_csv: str) -> pd.DataFrame:
    """
    Lee la base de medidores desde la URL CSV publicada de Google Sheets.
    """
    df = pd.read_csv(url_csv)
    return df


def mapear_modelo_corto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una columna 'Modelo_corto' a partir de:
      - Descripción larga (MAPA_DESCRIPCION_A_MODELO)
      - O usa directamente la descripción si ya es un modelo corto.
    """
    desc_original = df["Descripción"].astype(str).str.strip()

    modelos = []
    for d in desc_original:
        if d in MAPA_DESCRIPCION_A_MODELO:
            modelos.append(MAPA_DESCRIPCION_A_MODELO[d])
        elif d.upper() in MODELOS_VALIDOS:
            modelos.append(d.upper())
        else:
            modelos.append("SIN_MODELO")

    df = df.copy()
    df["Modelo_corto"] = modelos
    return df


# ============================================================
# 🧮 FUNCIÓN DE PREDICCIÓN COMPLETA
# ============================================================

def time_to_threshold(eq0: float, degr: float, emp: float, max_years: float) -> float:
    """
    Tiempo (años) para que |eq0 + degr * t| alcance emp (umbral).
    Devuelve el mínimo t >= 0. Si no cruza, devuelve max_years.
    """
    if abs(eq0) >= emp:
        return 0.0
    if degr == 0:
        return float(max_years)

    t1 = (emp - eq0) / degr
    t2 = (-emp - eq0) / degr
    candidates = [t for t in (t1, t2) if t >= 0]
    if not candidates:
        return float(max_years)
    return float(min(min(candidates), max_years))


def predecir_lote(df_input, interval_models, clasif_models, df_ic,
                  emp=EMP_Q3, max_years=MAX_YEARS):

    resultados = []

    # Aseguramos columnas necesarias
    df = df_input.copy()
    if "Estado" not in df.columns:
        df["Estado"] = "Usado"

    # Estado_Usado para el clasificador (USADO/NUEVO)
    df["Estado_Usado"] = df["Estado"].astype(str).str.upper().str.strip()
    df["Estado_Usado"] = df["Estado_Usado"].replace(
        {"USADO": "USADO", "NUEVO": "NUEVO"}
    )

    for _, row in df.iterrows():
        modelo = row["Modelo_corto"]
        edad = float(row.get("Edad", 0) or 0)
        volumen = float(row.get("Volumen", 0) or 0)
        consumo = float(row.get("Consumo_anual", 0) or 0)
        qmin = float(str(row.get("Qminimo", 0)).replace(",", ".") or 0)
        qtrans = float(str(row.get("Qtransicion", 0)).replace(",", ".") or 0)
        estado_u = str(row.get("Estado_Usado", "USADO")).strip().upper()

        # Si no hay modelos para este tipo:
        if modelo not in interval_models or modelo not in clasif_models:
            resultados.append({
                **row.to_dict(),
                "EQ3_lower (%)": np.nan,
                "EQ3_upper (%)": np.nan,
                "Pred_Conformidad": np.nan,
                "Probabilidad_Cumple": np.nan,
                "Nivel_riesgo": "SIN MODELO",
                "Vida_remanente_IC_inf (años, conservadora)": np.nan,
                "Dictamen_por_intervalo": "⚪ SIN MODELO"
            })
            continue

        # 1️⃣ Intervalo de error EQ3
        X_pred_int = pd.DataFrame([[edad, volumen, consumo]],
                                  columns=["Edad", "Volumen", "Consumo_anual"])

        eq3_low = float(interval_models[modelo]["lower"].predict(X_pred_int)[0])
        eq3_high = float(interval_models[modelo]["upper"].predict(X_pred_int)[0])
        eq3_mid = (eq3_low + eq3_high) / 2.0

        # 2️⃣ Clasificación de conformidad
        X_class = pd.DataFrame(
            [[edad, volumen, qmin, qtrans, estado_u]],
            columns=["Edad", "Volumen", "Qminimo", "Qtransicion", "Estado_Usado"]
        )
        pool = Pool(X_class, cat_features=["Estado_Usado"])
        prob_cumple = float(clasif_models[modelo].predict_proba(pool)[0, 1])
        pred_cumple = "CUMPLE" if prob_cumple >= 0.5 else "NO CUMPLE"

        # Nivel de riesgo
        if prob_cumple >= 0.80:
            nivel_riesgo = "Bajo"
        elif prob_cumple >= 0.50:
            nivel_riesgo = "Medio"
        else:
            nivel_riesgo = "Alto"

        # 3️⃣ Vida remanente (conservadora, usando degradación media por modelo)
        degr_series = df_ic.loc[df_ic["Descripción"] == modelo, "Degradación_media (%)"]
        degr = float(degr_series.values[0]) if len(degr_series) else -0.3  # fallback

        t_rem = time_to_threshold(eq3_mid, degr, emp, max_years)
        vida_rem = round(t_rem, 2)

        # Dictamen por intervalo
        if abs(eq3_mid) > emp:
            dictamen = "🔴 FUERA"
        else:
            dictamen = "🟢 DENTRO"

        resultados.append({
            **row.to_dict(),
            "EQ3_lower (%)": round(eq3_low, 3),
            "EQ3_upper (%)": round(eq3_high, 3),
            "Pred_Conformidad": pred_cumple,
            "Probabilidad_Cumple": round(prob_cumple, 3),
            "Nivel_riesgo": nivel_riesgo,
            "Vida_remanente_IC_inf (años, conservadora)": vida_rem,
            "Dictamen_por_intervalo": dictamen
        })

    return pd.DataFrame(resultados)


# ============================================================
# 🎛️ INTERFAZ STREAMLIT
# ============================================================

def main():
    st.set_page_config(
        page_title="Sistema de Monitoreo Metrológico",
        layout="wide"
    )

    st.title("📊 Sistema de Monitoreo Metrológico de Medidores de Agua")

    st.markdown("""
    Esta aplicación integra los modelos desarrollados en la Fase 3 del trabajo de grado:

    - Intervalos de error EQ3 (modelos cuantílicos por familia de medidor).
    - Clasificación de conformidad metrológica (CatBoost por modelo).
    - Estimación de vida útil remanente basada en degradación anual promedio por modelo.

    Todos los cálculos se realizan **a partir de la base maestra en Google Sheets**, sin necesidad
    de ejecutar código en Google Colab.
    """)

    st.sidebar.header("⚙️ Configuración de entrada")

    url_hoja = st.sidebar.text_input(
        "URL CSV publicada de Google Sheets",
        value=DEFAULT_SHEET_CSV_URL,
        help="Usa la URL generada en: Archivo → Compartir → Publicar en la web → CSV."
    )

    if st.sidebar.button("🔄 Cargar datos y ejecutar predicción"):
        if not url_hoja.strip():
            st.error("Por favor, proporciona la URL CSV publicada de tu Google Sheet.")
            return

        # 1) Cargar datos desde Sheets
        try:
            df_base = leer_base_medidores(url_hoja.strip())
        except Exception as e:
            st.error(f"Error al leer la hoja de cálculo: {e}")
            return

        st.success(f"Base cargada correctamente. Registros: {len(df_base)}")

        # 2) Mapear descripción → modelo corto
        if "Descripción" not in df_base.columns:
            st.error("La hoja debe contener una columna llamada 'Descripción'.")
            return

        df_base = mapear_modelo_corto(df_base)

        st.subheader("📋 Vista previa de la base de medidores")
        st.dataframe(df_base.head(20), use_container_width=True)

        # 3) Cargar modelos y degradaciones
        interval_models, clasif_models = cargar_modelos()
        df_ic = cargar_degradaciones()

        # 4) Ejecutar predicción
        with st.spinner("Ejecutando predicciones sobre la base de medidores..."):
            df_pred = predecir_lote(df_base, interval_models, clasif_models, df_ic)

        st.subheader("📈 Resultados de predicción")

        # Métricas resumen
        total = len(df_pred)
        con_modelo = (df_pred["Dictamen_por_intervalo"] != "⚪ SIN MODELO").sum()
        sin_modelo = total - con_modelo
        fuera = (df_pred["Dictamen_por_intervalo"] == "🔴 FUERA").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Medidores totales", total)
        col2.metric("Con modelo disponible", con_modelo)
        col3.metric("Fuera de intervalo (🔴)", fuera)

        st.dataframe(df_pred, use_container_width=True)

        # 5) Botón de descarga
        csv_out = df_pred.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Descargar resultados en CSV",
            data=csv_out,
            file_name="predicciones_medidores.csv",
            mime="text/csv"
        )

    else:
        st.info("Configura la URL de tu Google Sheets en el panel lateral y pulsa **'Cargar datos y ejecutar predicción'**.")


if __name__ == "__main__":
    main()
