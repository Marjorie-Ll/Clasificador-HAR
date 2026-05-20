import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(
    page_title="HAR Classifier",
    page_icon="🏃",
    layout="wide"
)

@st.cache_resource
def cargar_modelo():
    svm      = joblib.load("svm_app.pkl")
    scaler   = joblib.load("scaler_app.pkl")
    features = joblib.load("features_app.pkl")
    act_dict = joblib.load("activity_dict.pkl")
    return svm, scaler, features, act_dict

svm, scaler, features, activity_dict = cargar_modelo()

iconos = {
    "WALKING":            "🚶",
    "WALKING_UPSTAIRS":   "⬆️",
    "WALKING_DOWNSTAIRS": "⬇️",
    "SITTING":            "🪑",
    "STANDING":           "🧍",
    "LAYING":             "🛌"
}

colores = {
    "WALKING":            "#4ECDC4",
    "WALKING_UPSTAIRS":   "#45B7D1",
    "WALKING_DOWNSTAIRS": "#96CEB4",
    "SITTING":            "#FFEAA7",
    "STANDING":           "#DDA0DD",
    "LAYING":             "#98D8C8"
}

st.title("🏃 Human Activity Recognition")
st.markdown("Clasificador de actividades humanas desde señales de sensores de smartphone · Dataset UCI HAR")
st.markdown("---")

tab1, tab2 = st.tabs([
    "📊 Modelo original — SVM + PCA",
    "🔮 Modelo simplificado — Prueba aquí"
])

# ════════════════════════════════════════════════════════
# TAB 1 — Modelo original SVM + PCA
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("Rendimiento del modelo original")
    st.markdown("Modelo entrenado con **SVM kernel RBF** sobre **102 componentes PCA** que capturan el 95% de la varianza de 561 features.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "93.7%")
    with col2:
        st.metric("F1-Score", "93.8%")
    with col3:
        st.metric("Features originales", "561")
    with col4:
        st.metric("Componentes PCA", "102")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### F1-Score por actividad")
        actividades = ["LAYING", "WALKING", "WALKING_DOWNSTAIRS",
                       "WALKING_UPSTAIRS", "STANDING", "SITTING"]
        f1_scores   = [1.00, 0.96, 0.92, 0.92, 0.92, 0.90]

        fig = px.bar(
            x=f1_scores,
            y=actividades,
            orientation="h",
            color=actividades,
            color_discrete_sequence=list(colores.values()),
            text=[f"{v:.2f}" for v in f1_scores]
        )
        fig.update_layout(
            showlegend=False,
            xaxis_range=[0.85, 1.02],
            height=320,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="F1-Score",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Precision y Recall por actividad")
        df_metricas = pd.DataFrame({
            "Actividad": actividades,
            "Precision": [1.00, 0.95, 0.97, 0.89, 0.90, 0.93],
            "Recall":    [1.00, 0.97, 0.88, 0.94, 0.94, 0.88]
        })
        fig2 = px.bar(
            df_metricas.melt(
                id_vars="Actividad",
                var_name="Métrica",
                value_name="Valor"
            ),
            x="Actividad", y="Valor",
            color="Métrica",
            barmode="group",
            color_discrete_sequence=["#534AB7", "#4ECDC4"],
            text_auto=".2f"
        )
        fig2.update_layout(
            height=320,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-30,
            yaxis_range=[0.85, 1.02]
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.info("""
**¿Cómo funciona este modelo?**  
Entrenado con 561 features del dataset UCI HAR, reducidas a 102 componentes 
PCA conservando el 95% de la varianza. Alta precisión, pero requiere datos 
en el formato exacto del dataset original.

**¿Quieres probarlo?**  
Se desarrolló un modelo alternativo con las 130 features más importantes 
(de 566 = 561 originales + 5 nuevas por ingeniería de características), 
logrando 93% de accuracy, apenas 0.7% inferior. ¡Pruébalo en la pestaña siguiente!
    """)

    st.warning("""
**Versión demo** · En producción, la app recibiría datos del acelerómetro 
y giroscopio a 50Hz en ventanas de 2.56 segundos, calcularía automáticamente 
las 130 features y predecería la actividad en tiempo real.
    """)

# ════════════════════════════════════════════════════════
# TAB 2 — Modelo simplificado
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("Modelo simplificado — SVM con 130 features")
    st.markdown("""
Se reentrenó el clasificador usando las **130 features más importantes** 
según Random Forest (de 566 = 561 originales + 5 nuevas). 
Casi igual de preciso, pero mucho más fácil de usar.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "93.0%", delta="-0.7% vs original")
    with col2:
        st.metric("F1-Score", "92.9%")
    with col3:
        st.metric("Features", "130", delta="-431 features")

    st.markdown("---")
    st.markdown("#### Sube tu CSV para predecir")
    st.markdown("El CSV debe tener las 130 features del modelo. Usa el archivo `ejemplos_har.csv` que incluye una muestra por cada actividad.")

    archivo = st.file_uploader(
        "Sube un CSV con las 130 features",
        type=["csv"]
    )

    if archivo is not None:
        try:
            df_input = pd.read_csv(archivo)
            st.success(f"✅ CSV cargado: {len(df_input)} filas")

            faltantes = [f for f in features if f not in df_input.columns]
            if faltantes:
                st.error(f"❌ Faltan {len(faltantes)} features en el CSV.")
            else:
                
                X_input = pd.DataFrame(
                    df_input[features].values,
                    columns=features
                )
                X_scaled = scaler.transform(X_input)
                preds    = svm.predict(X_scaled)
                probs    = svm.predict_proba(X_scaled)

                df_input["Predicción"] = [activity_dict[p] for p in preds]

                st.markdown("---")
                st.markdown("#### Resultados")

                for i, row in df_input.iterrows():
                    pred  = row["Predicción"]
                    icono = iconos.get(pred, "❓")
                    color = colores.get(pred, "#E0E0E0")
                    prob  = probs[i].max()
                    real  = row.get("ActivityName", None)

                    col_a, col_b, col_c = st.columns([1, 2, 1])
                    with col_a:
                        st.markdown(f"**Fila {i+1}**")
                        if real:
                            st.markdown(f"Real: `{real}`")
                    with col_b:
                        st.markdown(
                            f"<div style='background:{color}; padding:12px; "
                            f"border-radius:10px; text-align:center; color:#1A1A4E;'>"
                            f"<h3 style='margin:0'>{icono} {pred}</h3>"
                            f"<p style='margin:4px 0 0; font-size:13px'>"
                            f"Confianza: {prob:.1%}</p></div>",
                            unsafe_allow_html=True
                        )
                    with col_c:
                        if real:
                            correcto = "✅" if real == pred else "❌"
                            st.markdown(f"### {correcto}")

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.markdown("""
        <div style='border:1.5px dashed #ccc; border-radius:10px; 
        padding:2rem; text-align:center; color:#888;'>
        <h3>📂 Arrastra tu CSV aquí</h3>
        <p>o usa el botón de arriba para seleccionar el archivo</p>
        </div>
        """, unsafe_allow_html=True)
