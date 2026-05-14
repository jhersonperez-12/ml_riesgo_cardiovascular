"""
app.py — Aplicación Streamlit para predicción de Riesgo Cardiovascular
Uso: streamlit run app.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Riesgo Cardiovascular",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CARGA DE ARTEFACTOS
# ──────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def cargar_modelo():
    modelo  = joblib.load(os.path.join(MODELS_DIR, "modelo_rf.joblib"))
    scaler  = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    le      = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    return modelo, scaler, le, feature_names


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
COLORES_RIESGO = {
    "ALTO":  ("#FF4B4B", "🔴"),
    "MEDIO": ("#FFA500", "🟠"),
    "BAJO":  ("#00C853", "🟢"),
}

def color_riesgo(clase: str):
    return COLORES_RIESGO.get(clase.upper(), ("#888888", "⚪"))


def preparar_fila(datos_usuario: dict, feature_names: list) -> pd.DataFrame:
    """
    Convierte el dict con los valores del usuario en un DataFrame
    con exactamente las columnas que espera el modelo
    (one-hot encoding + orden correcto).
    """
    fila = pd.DataFrame([datos_usuario])

    # One-hot encoding solo de categóricas (CATEGORIA_IMC)
    categoricas = fila.select_dtypes(include=["object"]).columns.tolist()
    fila = pd.get_dummies(fila, columns=categoricas, drop_first=True)

    # Alinear con las columnas del entrenamiento
    fila = fila.reindex(columns=feature_names, fill_value=0)
    return fila


# ──────────────────────────────────────────────
# INTERFAZ
# ──────────────────────────────────────────────
def main():
    # ── Cabecera ──────────────────────────────
    st.title("🫀 Clasificador de Riesgo Cardiovascular")
    st.markdown(
        "Ingresa los datos del paciente en el panel lateral y presiona **Predecir** "
        "para obtener la clasificación de riesgo cardiovascular."
    )
    st.divider()

    # ── Carga del modelo ──────────────────────
    try:
        modelo, scaler, le, feature_names = cargar_modelo()
    except FileNotFoundError:
        st.error(
            "⚠️ No se encontraron los artefactos del modelo. "
            "Ejecuta primero: `python src/train.py --data data/Clasificacion_RCV_Completo.xlsx`"
        )
        st.stop()

    # ── Sidebar — Datos del paciente ──────────
    st.sidebar.header("📋 Datos del paciente")

    with st.sidebar:

        st.subheader("Datos antropométricos")
        edad  = st.number_input("Edad (años)",  min_value=1,    max_value=120,   value=45,    step=1)
        peso  = st.number_input("Peso (kg)",    min_value=20.0, max_value=300.0, value=75.0,  step=0.1)
        talla = st.number_input("Talla (cm)",   min_value=50.0, max_value=250.0, value=170.0, step=0.1)
        imc   = st.number_input(
            "IMC (kg/m²)",
            min_value=10.0, max_value=80.0,
            value=round(peso / (talla / 100) ** 2, 1),
            step=0.1
        )
        cat_imc = st.selectbox(
            "Categoría IMC",
            # ⚠️ Verifica con: df["CATEGORIA_IMC"].unique()
            ["NORMAL", "SOBREPESO", "OBESIDAD", "BAJO PESO"]
        )

        st.subheader("Perfil lipídico")
        col_hdl   = st.number_input("Colesterol HDL (mg/dL)",   min_value=20.0,  max_value=200.0,  value=50.0,  step=0.1)
        col_ldl   = st.number_input("Colesterol LDL (mg/dL)",   min_value=20.0,  max_value=400.0,  value=100.0, step=0.1)
        trigli    = st.number_input("Triglicéridos (mg/dL)",    min_value=20.0,  max_value=1000.0, value=150.0, step=0.1)
        col_total = st.number_input("Colesterol total (mg/dL)", min_value=50.0,  max_value=500.0,  value=180.0, step=0.1)

        st.subheader("Presión arterial")
        pas = st.number_input("Presión sistólica (mmHg)",  min_value=60.0,  max_value=250.0, value=120.0, step=0.1)
        pad = st.number_input("Presión diastólica (mmHg)", min_value=40.0,  max_value=150.0, value=80.0,  step=0.1)

        st.subheader("Función renal")
        creat = st.number_input("Creatinina sérica (mg/dL)", min_value=0.1, max_value=20.0, value=1.0, step=0.01)

        st.subheader("Antecedentes")
        sexo       = st.selectbox("Sexo",             [0, 1], format_func=lambda x: "Femenino" if x == 0 else "Masculino")
        tabaquismo = st.selectbox(
            "Hábito tabáquico",
            [0, 1, 2, 3],
            format_func=lambda x: {0: "No fumador", 1: "Ex fumador", 2: "Fumador ocasional", 3: "Fumador habitual"}.get(x, str(x))
        )
        diabetes   = st.selectbox("Diabetes",         [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")

        predecir_btn = st.button("🔍 Predecir riesgo", use_container_width=True, type="primary")

    # ── Panel principal ───────────────────────
    col_resumen, col_detalle = st.columns([1, 2])

    with col_resumen:
        st.subheader("Resumen del paciente")
        resumen = {
            "Edad":              f"{edad} años",
            "Peso":              f"{peso} kg",
            "Talla":             f"{talla} cm",
            "IMC":               f"{imc:.1f} kg/m²",
            "Categoría IMC":     cat_imc,
            "Col. HDL":          f"{col_hdl:.1f} mg/dL",
            "Col. LDL":          f"{col_ldl:.1f} mg/dL",
            "Triglicéridos":     f"{trigli:.1f} mg/dL",
            "Col. Total":        f"{col_total:.1f} mg/dL",
            "P. Sistólica":      f"{pas:.1f} mmHg",
            "P. Diastólica":     f"{pad:.1f} mmHg",
            "Creatinina":        f"{creat:.2f} mg/dL",
            "Sexo":              "Masculino" if sexo == 1 else "Femenino",
            "Tabaquismo": {0: "No fumador", 1: "Ex fumador", 2: "Fumador ocasional", 3: "Fumador habitual"}.get(tabaquismo, str(tabaquismo)),
            "Diabetes":          "Sí" if diabetes == 1 else "No",
        }
        st.table(pd.DataFrame(resumen.items(), columns=["Variable", "Valor"]))

    with col_detalle:
        st.subheader("Resultado de la clasificación")

        if predecir_btn:
            datos_usuario = {
                "SEXO":                      sexo,
                "EDAD":                      edad,
                "PESO":                      peso,
                "TALLA":                     talla,
                "IMC":                       imc,
                "CATEGORIA_IMC":             cat_imc,
                "COLESTEROL_HDL":            col_hdl,
                "COLESTEROL_LDL":            col_ldl,
                "TRIGLICERIDOS":             trigli,
                "COLESTEROL_TOTAL":          col_total,
                "CREATININA_SERICA":         creat,
                "HABITO_TABAQUICO":          tabaquismo,
                "DIABETES":                  diabetes,
                "PRESION_ARTERIAL_SISTOLICA":  pas,
                "PRESION_ARTERIAL_DIASTOLICA": pad,
            }

            fila   = preparar_fila(datos_usuario, feature_names)
            fila_s = scaler.transform(fila)

            pred_num   = modelo.predict(fila_s)[0]
            pred_proba = modelo.predict_proba(fila_s)[0]
            pred_clase = le.inverse_transform([pred_num])[0]

            color, icono = color_riesgo(pred_clase)

            st.markdown(
                f"""
                <div style="
                    background:{color}22;
                    border: 2px solid {color};
                    border-radius: 12px;
                    padding: 20px 28px;
                    text-align: center;
                ">
                    <p style="font-size:3rem; margin:0">{icono}</p>
                    <p style="font-size:1.1rem; color:gray; margin:4px 0">Clasificación de riesgo</p>
                    <p style="font-size:2.5rem; font-weight:700; color:{color}; margin:0">
                        {pred_clase}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Probabilidades por clase")
            proba_df = pd.DataFrame(
                {"Clase": le.classes_, "Probabilidad (%)": (pred_proba * 100).round(2)}
            ).sort_values("Probabilidad (%)", ascending=False)

            for _, row in proba_df.iterrows():
                c, p = row["Clase"], row["Probabilidad (%)"]
                c_hex, _ = color_riesgo(c)
                st.markdown(f"**{c}**")
                st.progress(int(p), text=f"{p:.1f}%")

            st.markdown("#### Variables más influyentes (top 10)")
            importancias = pd.Series(
                modelo.feature_importances_, index=feature_names
            ).nlargest(10).sort_values()
            st.bar_chart(importancias)

        else:
            st.info("⬅️ Completa los datos en el panel lateral y presiona **Predecir riesgo**.")

    # ── Pie de página ─────────────────────────
    st.divider()
    st.caption(
        "⚠️ Esta herramienta es un apoyo computacional. "
        "No reemplaza el diagnóstico clínico de un profesional de la salud."
    )


if __name__ == "__main__":
    main()