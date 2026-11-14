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

# Rutas absolutas relativas al proyecto (evita que el logo o los JSON fallen)
BASE_DIR = Path(__file__).resolve().parent.parent
API_BASE = "http://127.0.0.1:8000"
MODELS_DIR = BASE_DIR / "models"
stats_path = MODELS_DIR / "dataset_stats.json"
history_path = MODELS_DIR / "training_history.json"
logo_path = BASE_DIR / "logo_optimax.png"

def load_json_data(path):
    """Carga segura de archivos JSON."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return None
    return None

stats_data = load_json_data(stats_path)
history_data = load_json_data(history_path)

# ==================== ESTILOS CSS (AQUÍ PUEDES JUGAR CON LOS COLORES) ====================
st.markdown("""
    <style>
    /* Reducir espacio superior desperdiciado */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1rem !important;
    }
    
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        height: 0px !important;
    }

    /* --- PALETA DE COLORES DEL ENCABEZADO (MODIFICA AQUÍ PARA PROBAR TONOS) --- */
    div[data-testid="stHorizontalBlock"]:has(#optimax-header-anchor) {
        background: linear-gradient(135deg, #071936 0%, #0B3B75 55%, #0077B6 100%) !important;
        border: 1.5px solid #00B4D8 !important;
        border-bottom: 3px solid #F97316 !important; /* Acento naranja del logo */
        border-radius: 14px !important;
        padding: 16px 22px !important;
        box-shadow: 0 6px 20px rgba(0, 180, 216, 0.25) !important;
        margin-bottom: 12px !important;
    }

    @media print {
        body {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        stButton, stFileUploader, .stSidebar {
            display: none !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ==================== ENCABEZADO CORPORATIVO ====================
col_logo, col_title, col_info = st.columns([1.1, 3.0, 1.4], vertical_alignment="center")

with col_logo:
    if logo_path.exists():
        st.image(str(logo_path), width=190)
    else:
        st.markdown("### 🫁 OPTIMAX")

with col_title:
    # Ancla invisible para que el CSS detecte exactamente esta fila y la pinte
    st.markdown("""
        <div id="optimax-header-anchor">
            <h2 style='margin:0; color:#FFFFFF; letter-spacing: 0.5px; text-shadow: 0 2px 4px rgba(0,0,0,0.4);'>
                OPTIMAX <span style='color:#38BDF8;'>|</span> Inteligencia Radiológica
            </h2>
            <p style='margin:4px 0 0 0; color:#E0F2FE; font-size:14px; font-weight: 500;'>
                Sistema Clínico Automatizado de Apoyo al Diagnóstico de Neumonía Pediátrica
            </p>
        </div>
    """, unsafe_allow_html=True)

with col_info:
    st.markdown("""
        <div style='background-color: rgba(4, 15, 32, 0.78); border: 1px solid #38BDF8; border-radius: 10px; padding: 10px 14px; color:#F0F9FF; font-size:12.5px; line-height:1.5;'>
            👤 <b>Dr. Asignado:</b> Clínica OPTIMAX<br>
            🌐 <b>Estado:</b> <span style='color:#4ADE80; font-weight:bold;'>Conectado a API REST</span><br>
            📞 <b>Soporte Tif:</b> Activo
        </div>
    """, unsafe_allow_html=True)

# ==================== PESTAÑAS DE NAVEGACIÓN ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Diagnosis & Grad-CAM",
    "⚙️ Entrenamiento y Parámetros",
    "📊 Métricas y Evaluación",
    "ℹ️ Arquitectura y Fuente"
])

# ==================== PESTAÑA 1: DIAGNÓSTICO (3 COLUMNAS) ====================
with tab1:
    col1, col2, col3 = st.columns([1.1, 1.2, 1.0], gap="medium")

    with col1:
        st.subheader("1. Carga de Imagen")
        uploaded_file = st.file_uploader("Subir radiografía (JPG, PNG)", type=["jpg", "jpeg", "png"])
        
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
        else:
            st.info("Cargue una radiografía para activar los controles.")

    with col2:
        st.subheader("2. Diagnóstico y Explicabilidad")
        threshold = st.slider("Umbral de Confianza Mínimo (%)", 50, 95, 70, 5)

        if uploaded_file is not None:
            if st.button("🔍 Analizar Radiografía", type="primary", use_container_width=True):
                with st.spinner("Analizando con CNN..."):
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
                        st.error("❌ Backend desconectado.")

            if "last_pred" in st.session_state:
                data = st.session_state["last_pred"]
                pred = data["prediction"]
                conf = data["confidence"]

                if conf < threshold:
                    st.warning(f"⚠️ CONFIANZA BAJA ({conf}% < {threshold}%): Revisión manual requerida.")
                elif pred == "Normal":
                    st.success(f"✅ NORMAL DETECTADO (Confianza: {conf}%)")
                else:
                    st.error(f"🚨 NEUMONÍA {pred.upper()} DETECTADA (Confianza: {conf}%)")

                if data.get("gradcam_base64"):
                    gcam_bytes = base64.b64decode(data["gradcam_base64"])
                    st.image(gcam_bytes, caption="Mapa de Calor Grad-CAM (Regiones de Activación)", use_container_width=True)
                else:
                    st.warning("Mapa de calor no disponible.")
        else:
            st.info("Esperando imagen para analizar...")

    with col3:
        st.subheader("3. Distribución de Probabilidades")
        if "last_pred" in st.session_state and uploaded_file is not None:
            data = st.session_state["last_pred"]
            pred = data["prediction"]
            probs = data["probabilities"]

            st.caption("Comparativa porcentual evaluada por MobileNetV2:")

            categories_info = [
                ("Normal", "🟢 Pulmón Sano", "#16A34A", "#052E16"),
                ("Bacterial", "🔴 Neumonía Bacteriana", "#DC2626", "#450A0A"),
                ("Viral", "🟠 Neumonía Viral", "#EA580C", "#431407")
            ]

            for cat_key, cat_title, border_color, bg_active in categories_info:
                val = probs.get(cat_key, 0.0)
                is_winner = (pred == cat_key)
                bg_color = bg_active if is_winner else "#1E293B"
                border_style = f"2px solid {border_color}" if is_winner else "1px solid #334155"
                badge = " ⭐ PRINCIPAL" if is_winner else "Evaluado"

                st.markdown(f"""
                <div style="background-color: {bg_color}; border: {border_style}; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 14px; color: #E2E8F0; font-weight: 600;">{cat_title}</span>
                        <span style="font-size: 11px; color: {border_color}; font-weight: 700;">{badge}</span>
                    </div>
                    <div style="font-size: 28px; font-weight: 800; color: #FFFFFF; margin-top: 6px;">{val:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Los porcentajes por patología aparecerán aquí tras ejecutar el análisis.")

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
            val_acc = history_data.get('val_accuracy', [0])
            last_acc = val_acc[-1] if val_acc else 0
            st.metric("Último Val Accuracy", f"{last_acc*100:.2f}%")
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
            ax1.plot(history_data.get("accuracy", []), label="Train Accuracy", marker="o")
            ax1.plot(history_data.get("val_accuracy", []), label="Val Accuracy", marker="s")
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
                    xticklabels=history_data.get("class_names", ["Bacterial", "Normal", "Viral"]),
                    yticklabels=history_data.get("class_names", ["Bacterial", "Normal", "Viral"]),
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