# DigitAI — Trackpad-Based Handwritten Digit Recognition Using Convolutional Neural Network

<div align="center">

**✦ DigitAI**

*Draw. Recognize. Understand.*

An interactive deep-learning web application that recognizes handwritten digits (0–9) drawn on a trackpad or mouse using a Convolutional Neural Network trained on the MNIST dataset.

---

</div>

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Features](#features)
5. [Technology Stack](#technology-stack)
6. [CNN Architecture](#cnn-architecture)
7. [Dataset](#dataset)
8. [Project Workflow](#project-workflow)
9. [Preprocessing Pipeline](#preprocessing-pipeline)
10. [Installation](#installation)
11. [Running the Application](#running-the-application)
12. [Project Structure](#project-structure)
13. [Results](#results)
14. [Future Scope](#future-scope)

---

## Project Overview

**Trackpad-Based Handwritten Digit Recognition Using Convolutional Neural Network** is a deep-learning application that demonstrates how neural networks can learn patterns from handwritten numerical data and classify previously unseen handwritten digits.

The system allows users to draw a single digit (0–9) on an interactive web canvas using a laptop trackpad or mouse. The drawing is captured, preprocessed into a 28 × 28 grayscale image matching the MNIST format, and classified by a trained CNN. The application displays the predicted digit, confidence score, probability distribution across all ten classes, and the processed input image.

**Core Pipeline:**

```
Trackpad Input → Digital Canvas → Image Preprocessing → CNN Model → Digit Prediction → Confidence Score
```

---

## Problem Statement

Handwritten digit recognition is a foundational problem in computer vision and deep learning. While automated systems can read printed text with high accuracy, recognizing handwritten digits remains challenging due to the natural variability in individual handwriting styles.

This project addresses the problem of classifying a **single handwritten digit** (0–9) provided via trackpad or mouse input, using a Convolutional Neural Network trained on the MNIST benchmark dataset.

---

## Objectives

1. **Design and train** a multi-layer Convolutional Neural Network for handwritten digit classification.
2. **Build an interactive web interface** that captures trackpad/mouse drawings on a digital canvas.
3. **Implement a robust preprocessing pipeline** that converts raw canvas input into MNIST-compatible format.
4. **Deploy the trained model** within a web application for real-time inference.
5. **Visualize the complete prediction pipeline** — from input to CNN output — for academic demonstration.

---

## Features

| Feature | Description |
|---|---|
| **Trackpad Drawing Canvas** | Interactive HTML5 canvas supporting mouse and trackpad input with smooth stroke rendering |
| **CNN Classification** | Multi-layer CNN trained on MNIST for accurate digit recognition |
| **Probability Distribution** | Visual display of softmax probabilities across all 10 digit classes |
| **Processed Input Preview** | 28 × 28 grayscale image showing exactly what the CNN receives |
| **Demo / Presentation Mode** | Side-by-side pipeline visualization for faculty demonstrations |
| **Model Architecture Visualization** | Interactive layer-by-layer architecture diagram with hover explanations |
| **Responsive Design** | Works on laptops, desktops, and tablets |
| **Error Handling** | Graceful handling of empty canvas, small drawings, and model unavailability |

---

## Technology Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **ML Framework** | TensorFlow / Keras |
| **Drawing Canvas** | streamlit-drawable-canvas |
| **Image Processing** | Pillow, NumPy |
| **Dataset** | MNIST |
| **Model Format** | `.keras` (TensorFlow SavedModel) |

---

## CNN Architecture

The Convolutional Neural Network follows a standard classification architecture with increasing filter depth:

```
Input Layer          28 × 28 × 1 (grayscale image)
        ↓
Data Augmentation    Random rotation, zoom, shift (training only)
        ↓
Conv2D               32 filters, 3 × 3 kernel
        ↓
Batch Normalization  Normalize activations
        ↓
Max Pooling          2 × 2 pool size
        ↓
Conv2D               64 filters, 3 × 3 kernel
        ↓
Batch Normalization  Normalize activations
        ↓
Max Pooling          2 × 2 pool size
        ↓
Conv2D               128 filters, 3 × 3 kernel
        ↓
Batch Normalization  Normalize activations
        ↓
Flatten              Reshape feature maps to 1D vector
        ↓
Dense                128 neurons, ReLU activation
        ↓
Dropout              Regularization
        ↓
Dense (Output)       10 neurons, Softmax activation
        ↓
Output               Probabilities for digits 0–9
```

### Training Configuration

| Parameter | Value |
|---|---|
| **Optimizer** | Adam |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Output Activation** | Softmax |
| **Input Shape** | (28, 28, 1) |
| **Number of Classes** | 10 |

---

## Dataset

The model is trained on the **MNIST** (Modified National Institute of Standards and Technology) handwritten digit dataset.

| Property | Value |
|---|---|
| **Training samples** | 60,000 |
| **Test samples** | 10,000 |
| **Image size** | 28 × 28 pixels |
| **Color** | Grayscale (single channel) |
| **Classes** | 10 (digits 0–9) |
| **Pixel range** | 0–255 (normalized to 0–1) |

---

## Project Workflow

```
User opens the website
        ↓
Navigates to "Recognizer" page
        ↓
Draws a digit (0–9) on the canvas using trackpad/mouse
        ↓
Clicks "Recognize Digit"
        ↓
System captures the canvas as an RGBA image
        ↓
Preprocessing pipeline:
  • Convert to grayscale
  • Detect digit bounding box
  • Crop with padding
  • Resize preserving aspect ratio
  • Center on 28 × 28 canvas
  • Normalize pixel values to [0, 1]
  • Reshape to (1, 28, 28, 1)
        ↓
Tensor fed into the trained CNN
        ↓
CNN produces 10-class softmax output
        ↓
Display:
  • Predicted digit (largest probability)
  • Confidence score (%)
  • Probability distribution (bar chart)
  • Processed 28 × 28 image
```

---

## Preprocessing Pipeline

The preprocessing module converts raw canvas drawings into CNN-compatible tensors. This step is critical because the model was trained on centered, normalized MNIST images.

```
Canvas RGBA image (H × W × 4)
        ↓
Extract grayscale via luminance
        ↓
Threshold to identify foreground pixels
        ↓
Find bounding box of the digit
        ↓
Validate minimum drawing size
        ↓
Crop with padding
        ↓
Resize preserving aspect ratio (fit within 20 × 20)
        ↓
Center on 28 × 28 canvas
        ↓
Normalize pixel values: [0, 255] → [0, 1]
        ↓
Reshape to tensor: (1, 28, 28, 1)
```

**Key Implementation Details:**

- The digit is fit into a 20 × 20 region and centered on a 28 × 28 canvas, following the original MNIST preprocessing methodology.
- Aspect ratio is preserved during resizing to avoid distortion.
- High-quality Lanczos resampling is used for resizing.
- Foreground/background polarity matches MNIST (white digits on black background).

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- A trained model file (`handwritten_digit_model.keras`)

### Steps

1. **Clone or download** the project:

   ```bash
   cd handwritten-digit-recognition
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Place your trained model:**

   Copy your `handwritten_digit_model.keras` file into the `model/` directory:

   ```
   model/handwritten_digit_model.keras
   ```

---

## Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### Quick Start

1. Click **"Try Recognizer"** or navigate to the **Recognizer** page.
2. Draw a digit (0–9) on the dark canvas using your trackpad or mouse.
3. Click **"Recognize Digit"**.
4. View the prediction, confidence score, probability distribution, and processed image.
5. Click **"Clear Canvas"** to try another digit.

### Demo Mode

Toggle the **🎬 Demo Mode** switch on the Recognizer page to display the preprocessing pipeline alongside the canvas — ideal for academic presentations.

---

## Project Structure

```
handwritten-digit-recognition/
│
├── app.py                          # Main Streamlit application
│
├── model/
│   ├── handwritten_digit_model.keras   # Trained CNN model (user-provided)
│   └── README.md                   # Model directory instructions
│
├── utils/
│   ├── __init__.py                 # Package initializer
│   ├── preprocessing.py            # Image preprocessing pipeline
│   └── prediction.py               # Model loading and inference
│
├── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
│
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `app.py` | UI rendering, page routing, navigation, CSS theming |
| `utils/preprocessing.py` | RGBA → grayscale, bounding box detection, crop, center, resize, normalize |
| `utils/prediction.py` | Model loading with caching, CNN inference, result formatting |

---

## Results

> **Note:** Prediction results displayed in the application are generated in real-time by the trained CNN model. No results are hardcoded or fabricated. The accuracy and confidence values depend entirely on the quality of the trained model and the clarity of the user's drawing.

The system displays:

- **Predicted Digit** — The digit class (0–9) with the highest softmax probability.
- **Confidence Score** — The softmax probability of the predicted class, expressed as a percentage.
- **Probability Distribution** — A horizontal bar chart showing the softmax output for all ten classes.
- **Processed Input** — The 28 × 28 normalized grayscale image fed to the CNN.

---

## Future Scope

1. **Multi-digit recognition** — Extend the system to recognize sequences of multiple digits.
2. **Real-time recognition** — Predict the digit as the user draws, without requiring a button click.
3. **Model comparison** — Allow users to switch between different trained models and compare predictions.
4. **Explainability** — Visualize CNN intermediate feature maps and Grad-CAM attention heatmaps.
5. **Extended character set** — Support alphabetic characters using the EMNIST dataset.
6. **Mobile optimization** — Improve touch drawing experience for mobile devices.
7. **Model retraining** — Allow users to contribute correctly labeled drawings to improve the model over time.

---

<div align="center">

**Built with** TensorFlow · Keras · Streamlit · MNIST

*Trackpad-Based Handwritten Digit Recognition Using Convolutional Neural Network*

</div>
