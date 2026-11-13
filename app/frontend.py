import streamlit as st
import requests
import json
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración de la página
st.set_page_config(
    page_title="Radiography AI - Clinical Triage",
    page_icon="🫁",
    layout="wide"
)

# URL del Backend FastAPI
API_URL = "http://127.0.0.1:8000/predict"

# Cargar estadísticas y métricas guardadas
MODELS_DIR = Path("models")
stats_path = MODELS_DIR / "dataset_stats.json"
history_path = MODELS_DIR / "training_history.json"

@st.cache_data
def load_json_data(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

stats_data = load_json_data(stats_path)
history_data = load_json_data(history_path)

# Estilo visual moderno y limpio
st.markdown("""
    <style>
    .main-title { font-size: 28px; font-weight: 700; color: #1E3A8A; }
    .sub-title { font-size: 16px; color: #64748B; }
    .metric-card { background-color: #F8FAFC; padding: 15px; border-radius: 10px; border: 1px solid #E2E8F0; }
    </style>
""", unsafe_allow_html=True)

# Encabezado principal
st.markdown('<p class="main-title">🫁 Radiography AI | Clinical Triage Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sistema inteligente de apoyo al diagnóstico médico de neumonía mediante Deep Learning</p>', unsafe_allow_html=True)
st.markdown("---")

# Pestañas de la aplicación (igual a tu diseño ideal)
tab1, tab2, tab3 = st.tabs(["🩺 Diagnosis", "📊 Dataset & Model Metrics", "ℹ️ Acerca del Proyecto"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### **Panel de Carga de Imagen**")
        uploaded_file = st.file_uploader("Seleccione una radiografía de tórax (JPEG, PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Radiografía de Tórax Cargada", use_column_width=True)
            
            # Controles de visualización (simulados para la interfaz)
            st.markdown("#### Controles de Visualización")
            zoom = st.slider("Zoom / Escala", 1.0, 2.0, 1.0)
            brightness = st.slider("Contraste / Brillo", 0.5, 1.5, 1.0)

    with col2:
        st.markdown("### **Resultado del Diagnóstico (IA)**")
        
        if uploaded_file is not None:
            if st.button("Ejecutar Diagnóstico Clínico", type="primary"):
                with st.spinner("Analizando patrones pulmonares con MobileNetV2..."):
                    try:
                        # Convertir imagen para enviar a la API
                        buffered = io.BytesIO()
                        image.save(buffered, format="JPEG")
                        buffered.seek(0)
                        
                        files = {"file": ("image.jpg", buffered, "image/jpeg")}
                        response = requests.post(API_URL, files=files)
                        
                        if response.status_code == 200:
                            res = response.json()
                            pred = res["prediction"]
                            conf = res["confidence"]
                            probs = res["probabilities"]
                            
                            # Alerta visual según el diagnóstico
                            if pred == "Normal":
                                st.success(f"✅ DIAGNÓSTICO: PULMONES NORMALES (Confianza: {conf}%)")
                            else:
                                st.error(f"⚠️ ALERTA: NEUMONÍA DETECTADA [{pred.upper()}] (Confianza: {conf}%)")
                            
                            st.markdown("#### Probabilidades por Categoría:")
                            for cat, p_val in probs.items():
                                st.progress(int(p_val), text=f"{cat}: {p_val}%")
                                
                        else:
                            st.error(f"Error en el servidor API: {response.text}")
                            
                    except requests.exceptions.ConnectionError:
                        st.error("No se pudo conectar con el Backend API. Asegúrate de ejecutar `uvicorn api.main:app --reload` en otra terminal.")
        else:
            st.info("👈 Sube una radiografía de tórax a la izquierda para comenzar el análisis en tiempo real.")

with tab2:
    st.markdown("### **Análisis Exploratorio y Métricas de Entrenamiento**")
    
    if stats_data and history_data:
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### Distribución de Clases (Dataset Real de Kaggle)")
            classes_df = pd.DataFrame(list(stats_data["classes"].items()), columns=["Categoría", "Cantidad"])
            st.bar_chart(classes_df.set_index("Categoría"))
            st.caption("Fuente bibliográfica: Paul Mooney (Chest X-Ray Images - Kaggle / Guangzhou Hospital)")

        with c2:
            st.markdown("#### Evolución del Accuracy durante el Entrenamiento")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(history_data["accuracy"], label="Train Accuracy", color="#2563EB", linewidth=2)
            ax.plot(history_data["val_accuracy"], label="Val Accuracy", color="#DC2626", linestyle="--", linewidth=2)
            ax.set_title("Precisión del Modelo por Época")
            ax.set_xlabel("Épocas")
            ax.set_ylabel("Accuracy")
            ax.legend()
            st.pyplot(fig)
            
    else:
        st.warning("⚠️ No se encontraron métricas guardadas. Ejecuta primero `python src/train.py`.")

with tab3:
    st.markdown("### **Información Académica del Proyecto**")
    st.markdown("""
    * **Asignatura:** Proyectos de Machine Learning (2026-2)
    * **Profesor:** Ing. Sergio Granizo MSc.
    * **Objetivo:** Desarrollar una aplicación funcional (Backend + Frontend en Python) aplicando inteligencia artificial para resolver un problema real de salud.
    * **Arquitectura:** 
        * *Modelo:* Transfer Learning con MobileNetV2 (Red Neuronal Convolucional).
        * *Backend:* FastAPI (API REST asíncrona).
        * *Frontend:* Streamlit (Dashboard interactivo).
    """)