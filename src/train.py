import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

IMG_SIZE = (224, 224)
TRAIN_DIR = Path("data_processed/train")
VAL_DIR = Path("data_processed/val")
TEST_DIR = Path("data_processed/test")
MODELS_DIR = Path("models")

def run_training(epochs: int = 5, learning_rate: float = 0.0001, batch_size: int = 32):
    print(f"🚀 Iniciando entrenamiento | Epochs: {epochs}, LR: {learning_rate}, Batch: {batch_size}")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=IMG_SIZE, batch_size=batch_size, label_mode="categorical", shuffle=True
    )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=IMG_SIZE, batch_size=batch_size, label_mode="categorical", shuffle=False
    )
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, image_size=IMG_SIZE, batch_size=batch_size, label_mode="categorical", shuffle=False
    )

    class_names = train_dataset.class_names

    # Normalización dentro del pipeline (0 a 1)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y)).prefetch(tf.data.AUTOTUNE)

    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(len(class_names), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

    # Evaluación final en el conjunto de Test
    y_true = []
    y_pred = []
    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "pneumonia_model.keras"
    model.save(model_path)

    metrics_data = {
        "hyperparameters": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "architecture": "MobileNetV2 (Transfer Learning)"
        },
        "accuracy": [float(v) for v in history.history['accuracy']],
        "val_accuracy": [float(v) for v in history.history['val_accuracy']],
        "loss": [float(v) for v in history.history['loss']],
        "val_loss": [float(v) for v in history.history['val_loss']],
        "class_names": class_names,
        "confusion_matrix": cm,
        "classification_report": report
    }

    history_path = MODELS_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)

    print("✅ Entrenamiento y evaluación finalizados.")
    return metrics_data

if __name__ == "__main__":
    run_training()