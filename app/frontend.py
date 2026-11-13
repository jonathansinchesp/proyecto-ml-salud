import streamlit as st
import requests
import json
import base64
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance, ImageOps
from pathlib import Path

# Configuración inicial de la página
st.set_page_config(page_title="OPTIMAX - Inteligencia Radiológica", page_icon="🫁", layout="wide")

API_BASE = "http://127.0.0.1:8000"
MODELS_DIR = Path("models")
stats_path = MODELS_DIR / "dataset_stats.json"
history_path = MODELS_DIR / "training_history.json"

def load_json_data(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

stats_data = load_json_data(stats_path)
history_data = load_json_data(history_path)

# ==================== ENCABEZADO CORPORATIVO (ESTILO IMAIOS) ====================
logo_path = Path("logo_optimax.png")

col_logo, col_title, col_info = st.columns([1.2, 3, 1.5])

with col_logo:
    if logo_path.exists():
        st.image(str(logo_path), width=220)
    else:
        st.markdown("### 🫁 OPTIMAX")

with col_title:
    st.markdown("## OPTIMAX | Inteligencia Radiológica")
    st.caption("Sistema Clínico Automatizado de Apoyo al Diagnóstico de Neumonía Pediátrica")

with col_info:
    st.markdown("👤 **Dr. Asignado:** Clínica OPTIMAX  \n🌐 **Estado:** Conectado a API REST  \n📞 **Soporte Tif:** Activo")

st.markdown("<hr style='margin: 5px 0px 15px 0px; border: 0.5px solid #334155;'>", unsafe_allow_html=True)

# ==================== PESTAÑAS DE NAVEGACIÓN ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Diagnosis & Grad-CAM",
    "⚙️ Entrenamiento y Parámetros",
    "📊 Métricas y Evaluación",
    "ℹ️ Arquitectura y Fuente"
])

# ==================== PESTAÑA 1: DIAGNÓSTICO ====================
with tab1:
    col1, col2 = st.columns([1.2, 1.4])

    with col1:
        st.subheader("1. Carga y Preprocesamiento")
        uploaded_file = st.file_uploader("Subir radiografía de tórax (JPG, PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            raw_image = Image.open(uploaded_file).convert("RGB")
            
            contrast = st.slider("Contraste Clínico", 0.5, 2.0, 1.0, 0.1)
            brightness = st.slider("Brillo", 0.5, 1.5, 1.0, 0.1)
            invert = st.toggle("Invertir Escala (Negativo)")

            proc_image = ImageEnhance.Contrast(raw_image).enhance(contrast)
            proc_image = ImageEnhance.Brightness(proc_image).enhance(brightness)
            if invert:
                proc_image = ImageOps.invert(proc_image)

            st.image(proc_image, caption="Vista Previa Preprocesada", use_container_width=True)

    with col2:
        st.subheader("2. Diagnóstico IA & Mapa de Calor")
        threshold = st.slider("Umbral de Confianza Mínimo (%)", 50, 95, 70, 5)

        if uploaded_file is not None:
            if st.button("🔍 Analizar Radiografía", type="primary", use_container_width=True):
                with st.spinner("Ejecutando inferencia y calculando Grad-CAM..."):
                    try:
                        buffered = io.BytesIO()
                        proc_image.save(buffered, format="JPEG")
                        buffered.seek(0)
                        
                        res = requests.post(f"{API_BASE}/predict", files={"file": ("xray.jpg", buffered, "image/jpeg")})
                        if res.status_code == 200:
                            st.session_state["last_pred"] = res.json()
                        else:
                            st.error(f"Error API: {res.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Backend desconectado. Ejecuta `uvicorn api.main:app --reload`")

            if "last_pred" in st.session_state:
                data = st.session_state["last_pred"]
                pred = data["prediction"]
                conf = data["confidence"]
                probs = data["probabilities"]

                if conf < threshold:
                    st.warning(f"⚠️ CONFIANZA BAJA ({conf}% < {threshold}%): Se recomienda revisión médica manual obligatoria.")
                elif pred == "Normal":
                    st.success(f"✅ NORMAL DETECTADO (Confianza: {conf}%)")
                else:
                    st.error(f"🚨 NEUMONÍA {pred.upper()} DETECTADA (Confianza: {conf}%)")

                # ==================== OPCIÓN 2: TARJETAS DINÁMICAS DE PROBABILIDAD ====================
                st.markdown("**Distribución de Probabilidades:**")
                card_cols = st.columns(3)
                
                categories_info = [
                    ("Normal", "🟢 Pulmón Sano", "#16A34A", "#052E16"),
                    ("Bacterial", "🔴 Neumonía Bacteriana", "#DC2626", "#450A0A"),
                    ("Viral", "🟠 Neumonía Viral", "#EA580C", "#431407")
                ]

                for idx, (cat_key, cat_title, border_color, bg_active) in enumerate(categories_info):
                    val = probs.get(cat_key, 0.0)
                    is_winner = (pred == cat_key)
                    bg_color = bg_active if is_winner else "#1E293B"
                    border_style = f"2px solid {border_color}" if is_winner else "1px solid #334155"
                    badge = " ⭐ PRINCIPAL" if is_winner else ""

                    with card_cols[idx]:
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; border: {border_style}; border-radius: 10px; padding: 12px; text-align: center; margin-bottom: 12px;">
                            <div style="font-size: 13px; color: #CBD5E1; font-weight: 600;">{cat_title}</div>
                            <div style="font-size: 26px; font-weight: 800; color: #FFFFFF; margin: 4px 0;">{val:.2f}%</div>
                            <div style="font-size: 11px; color: {border_color}; font-weight: 700;">{badge if badge else 'Evaluado'}</div>
                        </div>
                        """, unsafe_allow_html=True)
                # ======================================================================================

                if data.get("gradcam_base64"):
                    st.markdown("**Explicabilidad Visual (Grad-CAM):**")
                    gcam_bytes = base64.b64decode(data["gradcam_base64"])
                    st.image(gcam_bytes, caption="Regiones de activación pulmonar detectadas por la CNN", use_container_width=True)
        else:
            st.info("👈 Carga una imagen de rayos X en el panel izquierdo para realizar la predicción en tiempo real.")

# ==================== PESTAÑA 2: ENTRENAMIENTO ====================
with tab2:
    st.subheader("Configuración de Hiperparámetros y Entrenamiento en Vivo")
    st.write("Ajusta los parámetros de la red neuronal y entrena el modelo directamente a través de la API.")

    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        epochs_input = st.number_input("Número de Épocas (Epochs)", min_value=1, max_value=30, value=2)
    with c_p2:
        lr_input = st.selectbox("Tasa de Aprendizaje (Learning Rate)", [0.001, 0.0005, 0.0001, 0.00005], index=2)
    with c_p3:
        batch_input = st.selectbox("Tamaño de Lote (Batch Size)", [16, 32, 64], index=1)

    if st.button("🚀 Iniciar Entrenamiento del Modelo", type="primary"):
        with st.spinner(f"Entrenando MobileNetV2 por {epochs_input} épocas... Esto puede tardar unos minutos."):
            try:
                payload = {"epochs": int(epochs_input), "learning_rate": float(lr_input), "batch_size": int(batch_input)}
                response = requests.post(f"{API_BASE}/train", json=payload, timeout=1800)
                if response.status_code == 200:
                    st.success("🎉 ¡Entrenamiento completado y modelo actualizado en producción!")
                    st.rerun()
                else:
                    st.error(f"Error durante el entrenamiento: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend desconectado. Asegúrate de correr FastAPI primero.")

# ==================== PESTAÑA 3: MÉTRICAS Y FICHA TÉCNICA ====================
with tab3:
    st.subheader("Ficha Técnica del Modelo Activo (OPTIMAX AI)")
    
    if history_data:
        hp = history_data.get("hyperparameters", {})
        m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
        with m_c1:
            st.metric("Arquitectura", "MobileNetV2")
        with m_c2:
            st.metric("Último Val Accuracy", f"{history_data['val_accuracy'][-1]*100:.2f}%")
        with m_c3:
            st.metric("Épocas Entrenadas", hp.get('epochs', 'N/A'))
        with m_c4:
            st.metric("Learning Rate", hp.get('learning_rate', 'N/A'))
        with m_c5:
            st.metric("Batch Size", hp.get('batch_size', 'N/A'))
    else:
        st.warning("⚠️ Sin historial de entrenamiento disponible.")

    st.markdown("---")
    st.subheader("Métricas de Rendimiento y Validación")
    
    if stats_data and history_data:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Distribución de Clases en el Dataset**")
            df_classes = pd.DataFrame(list(stats_data["classes"].items()), columns=["Clase", "Imágenes"])
            st.bar_chart(df_classes.set_index("Clase"))

            st.markdown("**Curvas de Precisión (Accuracy)**")
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            ax1.plot(history_data["accuracy"], label="Train Accuracy", marker="o")
            ax1.plot(history_data["val_accuracy"], label="Val Accuracy", marker="s")
            ax1.set_xlabel("Época")
            ax1.set_ylabel("Accuracy")
            ax1.legend()
            st.pyplot(fig1)

        with col_m2:
            if "confusion_matrix" in history_data:
                st.markdown("**Matriz de Confusión (Conjunto Test)**")
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    history_data["confusion_matrix"],
                    annot=True, fmt="d", cmap="Blues",
                    xticklabels=history_data["class_names"],
                    yticklabels=history_data["class_names"],
                    ax=ax2
                )
                ax2.set_xlabel("Predicción")
                ax2.set_ylabel("Real")
                st.pyplot(fig2)

            if "classification_report" in history_data:
                st.markdown("**Reporte de Clasificación Detallado**")
                rep_df = pd.DataFrame(history_data["classification_report"]).transpose()
                st.dataframe(rep_df.style.format("{:.3f}"))
    else:
        st.info("Entrena el modelo al menos una vez para generar las métricas completas.")

# ==================== PESTAÑA 4: INFORMACIÓN ====================
with tab4:
    st.markdown("""
    ### Detalles Técnicos y Bibliográficos - OPTIMAX
    * **Institución:** Clínica Radiológica OPTIMAX
    * **Dataset Real:** *Chest X-Ray Images (Pneumonia)* - Kermany et al. (2018) / Repositorio Kaggle (Paul Mooney). Imágenes pediátricas reales de Guangzhou Women and Children's Medical Center.
    * **Preprocesamiento:** Estratificación 80% Train / 10% Validation / 10% Test, normalización de píxeles `[0,1]` y redimensionamiento a `224x224x3`.
    * **Modelo:** Transfer Learning con **MobileNetV2** + Capas Densas + Dropout + Explicabilidad mediante **Grad-CAM**.
    * **Stack Tecnológico:** Python, TensorFlow/Keras, FastAPI (Backend REST) y Streamlit (Frontend).
    """)