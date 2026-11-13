import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pathlib import Path

app = FastAPI(
    title="Radiography AI Backend",
    description="API de diagnóstico clínico de neumonía mediante Deep Learning",
    version="1.0.0"
)

# Cargar modelo entrenado al iniciar la API
MODEL_PATH = Path("models/pneumonia_model.keras")
model = None
CLASS_NAMES = ["Bacterial", "Normal", "Viral"]  # Orden alfabético por defecto de image_dataset_from_directory

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Modelo cargado correctamente en la API.")
    else:
        print("⚠️ Advertencia: No se encontró el modelo entrenado. Ejecuta src/train.py primero.")

@app.get("/")
def home():
    return {"status": "API activa", "message": "Bienvenido al backend de Neumonía AI"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado en el servidor.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo proporcionado no es una imagen válida.")

    try:
        # Leer y procesar imagen
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))
        
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Dimension batch (1, 224, 224, 3)

        # Predicción
        predictions = model.predict(img_array)
        scores = tf.nn.softmax(predictions[0]).numpy() if np.max(predictions[0]) > 1.0 else predictions[0]
        
        predicted_index = int(np.argmax(scores))
        confidence = float(scores[predicted_index]) * 100
        
        result_mapping = {i: float(scores[i]) * 100 for i in range(len(CLASS_NAMES))}

        return {
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": round(confidence, 2),
            "probabilities": {name: round(result_mapping[i], 2) for i, name in enumerate(CLASS_NAMES)}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {str(e)}")