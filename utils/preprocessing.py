"""
preprocessing.py
================
Image preprocessing pipeline for handwritten digit recognition.

Converts raw canvas RGBA data into a normalized 28×28 grayscale tensor
compatible with the CNN trained on MNIST.

Pipeline:
    Canvas RGBA → Grayscale → Threshold → Crop → Aspect-ratio-preserving
    resize → Center on 28×28 canvas → Normalize [0,1] → Shape (1,28,28,1)
"""

import numpy as np
from PIL import Image, ImageOps


def rgba_to_grayscale(image_array: np.ndarray) -> np.ndarray:
    """
    Convert an RGBA canvas image to a grayscale numpy array.
    The canvas has a dark background with bright strokes.
    We extract the alpha channel (or luminance) to isolate the drawing.

    Parameters
    ----------
    image_array : np.ndarray
        Raw RGBA image from the drawing canvas, shape (H, W, 4).

    Returns
    -------
    np.ndarray
        Grayscale image, shape (H, W), values in [0, 255].
    """
    if image_array.ndim == 2:
        return image_array

    if image_array.shape[2] == 4:
        # Use the alpha channel as the stroke indicator (canvas bg is transparent)
        alpha = image_array[:, :, 3]
        # Where alpha > 0, use the luminance of RGB; otherwise 0
        r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
        luminance = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
        gray = np.where(alpha > 0, luminance, 0).astype(np.uint8)
        return gray
    elif image_array.shape[2] == 3:
        r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
        gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
        return gray
    else:
        return image_array[:, :, 0]


def find_bounding_box(gray: np.ndarray, threshold: int = 20) -> tuple:
    """
    Find the bounding box of the drawn digit in the grayscale image.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image, shape (H, W).
    threshold : int
        Minimum pixel intensity to consider as foreground.

    Returns
    -------
    tuple
        (top, bottom, left, right) bounding box indices, or None if empty.
    """
    mask = gray > threshold
    if not mask.any():
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    return int(top), int(bottom), int(left), int(right)


def crop_and_center(gray: np.ndarray, bbox: tuple, target_size: int = 20,
                    canvas_size: int = 28, padding: int = 4) -> np.ndarray:
    """
    Crop the digit, resize preserving aspect ratio, and center on a
    28×28 canvas — mimicking MNIST preprocessing.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image with the drawn digit.
    bbox : tuple
        (top, bottom, left, right) bounding box of the digit.
    target_size : int
        Size to fit the digit into before centering (default 20 for MNIST).
    canvas_size : int
        Final canvas size (default 28).
    padding : int
        Padding around the digit in the bounding box.

    Returns
    -------
    np.ndarray
        Processed image, shape (28, 28), values in [0, 255].
    """
    top, bottom, left, right = bbox

    # Add padding around the bounding box
    h, w = gray.shape
    top = max(0, top - padding)
    bottom = min(h - 1, bottom + padding)
    left = max(0, left - padding)
    right = min(w - 1, right + padding)

    # Crop the digit
    cropped = gray[top:bottom + 1, left:right + 1]

    # Convert to PIL for high-quality resize
    pil_img = Image.fromarray(cropped)

    # Resize preserving aspect ratio to fit within target_size × target_size
    crop_h, crop_w = cropped.shape
    if crop_h > crop_w:
        new_h = target_size
        new_w = max(1, int(crop_w * (target_size / crop_h)))
    else:
        new_w = target_size
        new_h = max(1, int(crop_h * (target_size / crop_w)))

    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

    # Create the final canvas and center the digit
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    offset_y = (canvas_size - new_h) // 2
    offset_x = (canvas_size - new_w) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = np.array(pil_img)

    return canvas


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to [0, 1] range and reshape for CNN input.

    Parameters
    ----------
    image : np.ndarray
        28×28 grayscale image, values in [0, 255].

    Returns
    -------
    np.ndarray
        Normalized tensor, shape (1, 28, 28, 1).
    """
    normalized = image.astype(np.float32) / 255.0
    return normalized.reshape(1, 28, 28, 1)


def preprocess_canvas(image_array: np.ndarray) -> dict:
    """
    Full preprocessing pipeline: Canvas RGBA → CNN-ready tensor.

    Parameters
    ----------
    image_array : np.ndarray
        Raw RGBA canvas image.

    Returns
    -------
    dict
        {
            'tensor': np.ndarray of shape (1, 28, 28, 1) or None,
            'processed_image': np.ndarray of shape (28, 28) or None,
            'status': str ('success', 'empty', 'too_small'),
            'message': str
        }
    """
    # Step 1: Convert to grayscale
    gray = rgba_to_grayscale(image_array)

    # Step 2: Find bounding box
    bbox = find_bounding_box(gray)

    if bbox is None:
        return {
            'tensor': None,
            'processed_image': None,
            'status': 'empty',
            'message': 'Please draw a digit before recognition.'
        }

    top, bottom, left, right = bbox
    digit_height = bottom - top
    digit_width = right - left

    # Step 3: Check if drawing is too small
    min_dimension = 10
    if digit_height < min_dimension and digit_width < min_dimension:
        return {
            'tensor': None,
            'processed_image': None,
            'status': 'too_small',
            'message': 'Please draw the digit more clearly.'
        }

    # Step 4: Crop, resize, center
    processed = crop_and_center(gray, bbox)

    # Step 5: Normalize for CNN
    tensor = normalize_image(processed)

    return {
        'tensor': tensor,
        'processed_image': processed,
        'status': 'success',
        'message': 'Image preprocessed successfully.'
    }
