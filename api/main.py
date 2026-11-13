import io
import base64
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
from pathlib import Path
from src.train import run_training

app = FastAPI(
    title="Radiography AI Backend",
    description="API de entrenamiento y diagnóstico clínico de neumonía",
    version="2.0.0"
)

MODEL_PATH = Path("models/pneumonia_model.keras")
model = None
CLASS_NAMES = ["Bacterial", "Normal", "Viral"]

class TrainRequest(BaseModel):
    epochs: int = 3
    learning_rate: float = 0.0001
    batch_size: int = 32

def load_current_model():
    global model
    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Modelo cargado en memoria.")

@app.on_event("startup")
def startup_event():
    load_current_model()

def generate_gradcam(img_array, model, pred_index):
    """Genera mapa de calor Grad-CAM sobre la última capa convolucional de MobileNetV2."""
    try:
        last_conv_layer_name = "out_relu"
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Superponer mapa de calor sobre la imagen original
        orig_img = np.uint8(255 * img_array[0])
        heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)
        pil_img = Image.fromarray(superimposed_img)
        
        buff = io.BytesIO()
        pil_img.save(buff, format="PNG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")
    except Exception:
        return None

@app.get("/")
def home():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/train")
def train_endpoint(params: TrainRequest):
    """Entrena el modelo desde la API con hiperparámetros configurables."""
    try:
        metrics = run_training(
            epochs=params.epochs,
            learning_rate=params.learning_rate,
            batch_size=params.batch_size
        )
        load_current_model()
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no está cargado. Entrénalo primero.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_resized = image.resize((224, 224))
        
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_batch, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index]) * 100

        gradcam_b64 = generate_gradcam(img_batch, model, predicted_index)

        return {
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": round(confidence, 2),
            "probabilities": {name: round(float(predictions[i]) * 100, 2) for i, name in enumerate(CLASS_NAMES)},
            "gradcam_base64": gradcam_b64
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")