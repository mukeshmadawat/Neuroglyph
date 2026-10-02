"""
train_model.py
==============
Trains the CNN model on the MNIST dataset, evaluates empirical test metrics,
generates the 10x10 confusion matrix, classification report, training curves,
and saves the trained model to handwritten_digit_model.keras.

Run:
    python train_model.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report

print("TensorFlow version:", tf.__version__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "handwritten_digit_model.keras")
ROOT_MODEL_PATH = os.path.join(ROOT_DIR, "handwritten_digit_model.keras")

# ============================================================
# 1. LOAD MNIST DATASET
# ============================================================
print("\nLoading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training images:", x_train.shape)
print("Testing images :", x_test.shape)

# ============================================================
# 2. PREPROCESS DATA
# ============================================================
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("\nAfter preprocessing:")
print("Training:", x_train.shape)
print("Testing :", x_test.shape)

# ============================================================
# 3. DATA AUGMENTATION
# ============================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(
        height_factor=0.10,
        width_factor=0.10
    )
], name="data_augmentation")

# ============================================================
# 4. BUILD CNN MODEL
# ============================================================
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    data_augmentation,
    # First convolution block
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # Second convolution block
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    # Third convolution block
    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    # Classifier head
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
], name="DigitCNN")

# ============================================================
# 5. DISPLAY MODEL
# ============================================================
model.summary()

# ============================================================
# 6. COMPILE MODEL
# ============================================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================================
# 7. CALLBACKS
# ============================================================
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001,
    verbose=1
)

# ============================================================
# 8. TRAIN MODEL
# ============================================================
print("\nStarting training...\n")
history = model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ============================================================
# 9. EVALUATE MODEL
# ============================================================
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# ============================================================
# 10. TRAINING ACCURACY & LOSS GRAPHS
# ============================================================
# Accuracy Graph
plt.figure(figsize=(8, 5), dpi=150)
plt.plot(history.history["accuracy"], label="Training Accuracy", color="#3b82f6", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="#10b981", linewidth=2)
plt.title("Training and Validation Accuracy", fontsize=12, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "accuracy_curve.png"), dpi=150)
plt.close()

# Loss Graph
plt.figure(figsize=(8, 5), dpi=150)
plt.plot(history.history["loss"], label="Training Loss", color="#ef4444", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", color="#f59e0b", linewidth=2)
plt.title("Training and Validation Loss", fontsize=12, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "loss_curve.png"), dpi=150)
plt.close()

# ============================================================
# 11. PREDICTIONS & CONFUSION MATRIX
# ============================================================
print("\nGenerating predictions on 10,000 test images...")
predictions = model.predict(x_test, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")
report_text = classification_report(y_test, predicted_labels)
report_dict = classification_report(y_test, predicted_labels, output_dict=True)
print(report_text)

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(9, 7), dpi=150)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)])
plt.title("MNIST Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
plt.xlabel("Predicted Digit", fontsize=11, labelpad=8)
plt.ylabel("Actual Digit", fontsize=11, labelpad=8)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

# Sample predictions plot
plt.figure(figsize=(12, 8), dpi=150)
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    correct = (y_test[i] == predicted_labels[i])
    col = "green" if correct else "red"
    plt.title(f"True: {y_test[i]} | Pred: {predicted_labels[i]}", fontsize=9, color=col, fontweight="bold")
    plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "sample_predictions.png"), dpi=150)
plt.close()

# Save Metrics JSON
metrics_data = {
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "test_samples": int(len(y_test)),
    "num_classes": 10,
    "confusion_matrix": cm.tolist(),
    "history": {
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "epochs": list(range(1, len(history.history["accuracy"]) + 1))
    },
    "classification_report_text": report_text,
    "classification_report_dict": report_dict
}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=2)

# ============================================================
# 12. SAVE MODEL
# ============================================================
model.save(ROOT_MODEL_PATH)
model.save(MODEL_PATH)

print("\n================================")
print("MODEL SAVED SUCCESSFULLY")
print("================================")
print(f"Saved to: {ROOT_MODEL_PATH}")
print(f"Saved to: {MODEL_PATH}")
print("\nYou can now launch the web app: streamlit run app.py")
