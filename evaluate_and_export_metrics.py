"""
evaluate_and_export_metrics.py
==============================
Evaluates the trained model on the 10,000 MNIST test images, computes:
1. Test loss and test accuracy
2. 10x10 Confusion Matrix (via scikit-learn & seaborn)
3. Full classification report
4. Training and validation loss & accuracy curves (parsed from task-29.log)
5. Sample test predictions
Saves all metrics and charts into the `model/` directory for the Streamlit Results page.
"""

import os
import re
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "handwritten_digit_model.keras")
ROOT_MODEL_PATH = os.path.join(ROOT_DIR, "handwritten_digit_model.keras")

# Copy model to root folder as well
if os.path.exists(MODEL_PATH) and not os.path.exists(ROOT_MODEL_PATH):
    shutil.copy2(MODEL_PATH, ROOT_MODEL_PATH)

print("Loading model and MNIST test data...")
model = tf.keras.models.load_model(MODEL_PATH)

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test_norm = x_test.astype("float32") / 255.0
x_test_norm = np.expand_dims(x_test_norm, axis=-1)

# Evaluate
test_loss, test_accuracy = model.evaluate(x_test_norm, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%")

# Predict all
print("Generating predictions on 10,000 test images...")
preds = model.predict(x_test_norm, batch_size=256, verbose=0)
pred_labels = np.argmax(preds, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, pred_labels)
print("Confusion Matrix calculated.")

# Classification Report
clf_report_dict = classification_report(y_test, pred_labels, output_dict=True)
clf_report_text = classification_report(y_test, pred_labels)

# Generate Confusion Matrix Plot
plt.figure(figsize=(8, 6), dpi=150)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=[str(i) for i in range(10)],
            yticklabels=[str(i) for i in range(10)])
plt.title("MNIST Confusion Matrix (10,000 Test Images)", fontsize=13, pad=12, fontweight='bold')
plt.xlabel("Predicted Digit", fontsize=11, labelpad=8)
plt.ylabel("Actual Digit", fontsize=11, labelpad=8)
plt.tight_layout()
cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()

# Parse task-29.log to extract real epoch-by-epoch history
log_candidates = [
    r"C:\Users\mukes\.gemini\antigravity-ide\brain\3020e7c1-3bca-4591-8faf-92f2c434947b\.system_generated\tasks\task-29.log"
]
history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": [], "epochs": []}
for log_file in log_candidates:
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            # Match lines like: 422/422 ... - accuracy: 0.9652 - loss: 0.1179 - val_accuracy: 0.9853 - val_loss: 0.0539
            pattern = r"accuracy:\s*([0-9\.]+)\s*-\s*loss:\s*([0-9\.]+)\s*-\s*val_accuracy:\s*([0-9\.]+)\s*-\s*val_loss:\s*([0-9\.]+)"
            matches = re.findall(pattern, content)
            if matches:
                for idx, (acc, loss, val_acc, val_loss) in enumerate(matches, 1):
                    history["epochs"].append(idx)
                    history["accuracy"].append(float(acc))
                    history["loss"].append(float(loss))
                    history["val_accuracy"].append(float(val_acc))
                    history["val_loss"].append(float(val_loss))
                break

# Plot Accuracy Curve
if history["epochs"]:
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(history["epochs"], history["accuracy"], label="Training Accuracy", color="#3b82f6", linewidth=2.2, marker='o', markersize=4)
    plt.plot(history["epochs"], history["val_accuracy"], label="Validation Accuracy", color="#10b981", linewidth=2.2, marker='s', markersize=4)
    plt.title("Training and Validation Accuracy", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim([min(history["accuracy"]) - 0.02, 1.005])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    acc_curve_path = os.path.join(MODEL_DIR, "accuracy_curve.png")
    plt.savefig(acc_curve_path, dpi=150)
    plt.close()

    # Plot Loss Curve
    plt.figure(figsize=(8, 4.5), dpi=150)
    plt.plot(history["epochs"], history["loss"], label="Training Loss", color="#ef4444", linewidth=2.2, marker='o', markersize=4)
    plt.plot(history["epochs"], history["val_loss"], label="Validation Loss", color="#f59e0b", linewidth=2.2, marker='s', markersize=4)
    plt.title("Training and Validation Loss", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    loss_curve_path = os.path.join(MODEL_DIR, "loss_curve.png")
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()

# Save sample predictions plot (20 test samples)
plt.figure(figsize=(12, 6.5), dpi=150)
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    is_correct = (y_test[i] == pred_labels[i])
    color = "green" if is_correct else "red"
    plt.title(f"True: {y_test[i]} | Pred: {pred_labels[i]}", fontsize=9, color=color, fontweight='bold')
    plt.axis("off")
plt.tight_layout()
samples_path = os.path.join(MODEL_DIR, "sample_predictions.png")
plt.savefig(samples_path, dpi=150)
plt.close()

# Save metrics JSON
metrics_data = {
    "test_accuracy": float(test_accuracy),
    "test_loss": float(test_loss),
    "test_samples": int(len(y_test)),
    "num_classes": 10,
    "confusion_matrix": cm.tolist(),
    "history": history,
    "classification_report_text": clf_report_text,
    "classification_report_dict": clf_report_dict,
}

metrics_json_path = os.path.join(MODEL_DIR, "metrics.json")
with open(metrics_json_path, "w", encoding="utf-8") as f:
    json.dump(metrics_data, f, indent=2)

print(f"Metrics saved to {metrics_json_path}")
print("All charts and metrics successfully generated!")
