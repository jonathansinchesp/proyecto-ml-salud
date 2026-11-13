import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from pathlib import Path

# Configuración de hiperparámetros configurables
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # Puedes subirlo a 10 o 15 si tienes tarjeta gráfica (GPU)
LEARNING_RATE = 0.0001

TRAIN_DIR = Path("data_processed/train")
VAL_DIR = Path("data_processed/val")
MODELS_DIR = Path("models")

def train_model():
    print("🚀 Cargando datasets con generadores de TensorFlow...")
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_dataset.class_names
    print(f"🏷️ Clases detectadas: {class_names}")

    # Optimización de rendimiento con prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

    print("🧠 Construyendo modelo basado en Transfer Learning (MobileNetV2)...")
    # Base preentrenada en ImageNet
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Congelar capas base inicialmente

    # Capas personalizadas para nuestras 3 clases de neumonía
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"🏋️ Iniciando entrenamiento por {EPOCHS} épocas...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS
    )

    # Guardar modelo entrenado
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "pneumonia_model.keras"
    model.save(model_path)
    print(f"💾 Modelo guardado exitosamente en: {model_path}")

    # Guardar métricas de entrenamiento para el dashboard
    history_serializable = {
        "accuracy": [float(val) for val in history.history['accuracy']],
        "val_accuracy": [float(val) for val in history.history['val_accuracy']],
        "loss": [float(val) for val in history.history['loss']],
        "val_loss": [float(val) for val in history.history['val_loss']],
        "class_names": class_names
    }

    history_path = MODELS_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_serializable, f, indent=4)
    print(f"📈 Métricas de entrenamiento guardadas en: {history_path}")

if __name__ == "__main__":
    train_model()