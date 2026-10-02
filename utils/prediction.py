"""
prediction.py
=============
Model loading, caching, and prediction utilities.

Loads the trained Keras CNN model once and caches it for repeated predictions.
Provides a clean interface for digit classification.
"""

import os
import numpy as np

# Suppress TensorFlow verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st


# ──────────────────────────────────────────────────
# Model path configuration
# ──────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
MODEL_FILENAME = 'handwritten_digit_model.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


@st.cache_resource(show_spinner=False)
def _load_keras_model(filepath: str, mtime: float):
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    return tf.keras.models.load_model(filepath)


def load_model():
    """
    Load the trained Keras CNN model and cache it across Streamlit sessions.

    Returns
    -------
    model or None
        The loaded Keras model, or None if loading fails.
    error : str or None
        Error message if loading failed, else None.
    """
    if not os.path.exists(MODEL_PATH):
        return None, (
            f"Model file not found at: {MODEL_PATH}\n\n"
            f"Please wait a moment if training is currently running, "
            f"or ensure `{MODEL_FILENAME}` is placed in the `model/` directory."
        )

    try:
        mtime = os.path.getmtime(MODEL_PATH)
        model = _load_keras_model(MODEL_PATH, mtime)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {str(e)}"



def predict_digit(model, tensor: np.ndarray) -> dict:
    """
    Run inference on the preprocessed image tensor.

    Parameters
    ----------
    model : keras.Model
        The loaded CNN model.
    tensor : np.ndarray
        Preprocessed image tensor, shape (1, 28, 28, 1).

    Returns
    -------
    dict
        {
            'predicted_digit': int,
            'confidence': float (0–100),
            'probabilities': np.ndarray of shape (10,),
            'status': str ('success' or 'error'),
            'message': str
        }
    """
    try:
        # Run prediction
        probabilities = model.predict(tensor, verbose=0)
        probs = probabilities[0]  # Shape: (10,)

        predicted_digit = int(np.argmax(probs))
        confidence = float(probs[predicted_digit]) * 100.0

        return {
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'probabilities': probs,
            'status': 'success',
            'message': 'Digit recognized'
        }

    except Exception as e:
        return {
            'predicted_digit': None,
            'confidence': None,
            'probabilities': None,
            'status': 'error',
            'message': f"Prediction failed: {str(e)}"
        }
