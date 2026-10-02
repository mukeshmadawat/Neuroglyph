"""
gui_tkinter.py — Desktop Handwritten Digit Recognition with Tkinter
===================================================================
A standalone desktop application allowing users to draw digits on a canvas
and predict them using the trained CNN model.

Run:
    python gui_tkinter.py
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "model", "handwritten_digit_model.keras")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(ROOT_DIR, "handwritten_digit_model.keras")

class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognition — Tkinter & CNN")
        self.root.geometry("640x520")
        self.root.configure(bg="#0f172a")
        self.root.resizable(False, False)

        # Load Model
        self.model = None
        self.load_deep_model()

        # Canvas Dimensions
        self.canvas_size = 280
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

        self.setup_ui()

    def load_deep_model(self):
        try:
            if os.path.exists(MODEL_PATH):
                self.model = tf.keras.models.load_model(MODEL_PATH)
            else:
                print("Model file not found at:", MODEL_PATH)
        except Exception as e:
            print("Error loading model:", e)

    def setup_ui(self):
        # Header
        header = tk.Label(
            self.root,
            text="Handwritten Digit Recognition",
            font=("Helvetica", 18, "bold"),
            fg="#f8fafc",
            bg="#0f172a",
            pady=10
        )
        header.pack()

        subtitle = tk.Label(
            self.root,
            text="Draw a digit (0 to 9) in the box below",
            font=("Helvetica", 10),
            fg="#94a3b8",
            bg="#0f172a"
        )
        subtitle.pack()

        # Main Container
        main_frame = tk.Frame(self.root, bg="#0f172a", pady=15)
        main_frame.pack()

        # Left Column: Canvas
        left_frame = tk.Frame(main_frame, bg="#0f172a", padx=20)
        left_frame.grid(row=0, column=0)

        self.canvas = tk.Canvas(
            left_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            cursor="crosshair",
            highlightthickness=2,
            highlightbackground="#6366f1"
        )
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset_coordinates)

        # Canvas Buttons
        btn_frame = tk.Frame(left_frame, bg="#0f172a", pady=12)
        btn_frame.pack(fill="x")

        clear_btn = tk.Button(
            btn_frame,
            text="Clear",
            font=("Helvetica", 11, "bold"),
            bg="#334155",
            fg="white",
            activebackground="#475569",
            activeforeground="white",
            relief="flat",
            padx=15,
            pady=6,
            command=self.clear_canvas
        )
        clear_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))

        recognize_btn = tk.Button(
            btn_frame,
            text="Recognize Digit",
            font=("Helvetica", 11, "bold"),
            bg="#6366f1",
            fg="white",
            activebackground="#4f46e5",
            activeforeground="white",
            relief="flat",
            padx=15,
            pady=6,
            command=self.predict
        )
        recognize_btn.pack(side="right", expand=True, fill="x", padx=(6, 0))

        # Right Column: Prediction Results
        right_frame = tk.Frame(main_frame, bg="#1e293b", padx=20, pady=15, width=220)
        right_frame.grid(row=0, column=1, sticky="nsew")

        pred_title = tk.Label(
            right_frame,
            text="PREDICTION",
            font=("Helvetica", 10, "bold"),
            fg="#818cf8",
            bg="#1e293b"
        )
        pred_title.pack(anchor="center", pady=(5, 0))

        self.digit_label = tk.Label(
            right_frame,
            text="—",
            font=("Helvetica", 64, "bold"),
            fg="#ffffff",
            bg="#1e293b"
        )
        self.digit_label.pack(anchor="center")

        self.conf_label = tk.Label(
            right_frame,
            text="Confidence: —%",
            font=("Helvetica", 11, "bold"),
            fg="#34d399",
            bg="#1e293b"
        )
        self.conf_label.pack(anchor="center", pady=(0, 15))

        # Probability distribution text
        prob_header = tk.Label(
            right_frame,
            text="Probabilities (0–9)",
            font=("Helvetica", 9, "bold"),
            fg="#94a3b8",
            bg="#1e293b"
        )
        prob_header.pack(anchor="w")

        self.prob_text = tk.Label(
            right_frame,
            text="\n".join([f"{i}:  0.0%" for i in range(10)]),
            font=("Consolas", 8),
            fg="#cbd5e1",
            bg="#0f172a",
            justify="left",
            padx=8,
            pady=6,
            relief="flat"
        )
        self.prob_text.pack(fill="x", pady=5)

    def paint(self, event):
        brush_r = 12
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=brush_r*2, fill="white", capstyle=tk.ROUND, smooth=True
            )
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=255, width=brush_r*2
            )
        else:
            self.canvas.create_oval(
                x - brush_r, y - brush_r, x + brush_r, y + brush_r,
                fill="white", outline="white"
            )
            self.draw.ellipse(
                [x - brush_r, y - brush_r, x + brush_r, y + brush_r],
                fill=255, outline=255
            )
        self.last_x, self.last_y = x, y

    def reset_coordinates(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.digit_label.config(text="—")
        self.conf_label.config(text="Confidence: —%")
        self.prob_text.config(text="\n".join([f"{i}:  0.0%" for i in range(10)]))

    def predict(self):
        if self.model is None:
            messagebox.showerror("Error", "Trained model not found. Please verify handwritten_digit_model.keras.")
            return

        # Preprocess PIL image
        bbox = self.image.getbbox()
        if not bbox:
            messagebox.showinfo("Canvas Empty", "Please draw a digit first.")
            return

        cropped = self.image.crop(bbox)
        # Pad to square
        w, h = cropped.size
        max_dim = max(w, h)
        square = Image.new("L", (max_dim, max_dim), color=0)
        square.paste(cropped, ((max_dim - w) // 2, (max_dim - h) // 2))

        # Resize with padding
        resized = square.resize((20, 20), Image.Resampling.LANCZOS)
        final_img = Image.new("L", (28, 28), color=0)
        final_img.paste(resized, (4, 4))

        # Array & Normalization
        arr = np.array(final_img, dtype=np.float32) / 255.0
        tensor = arr.reshape(1, 28, 28, 1)

        # Inference
        probs = self.model.predict(tensor, verbose=0)[0]
        predicted_digit = int(np.argmax(probs))
        confidence = float(probs[predicted_digit]) * 100.0

        # Update UI
        self.digit_label.config(text=str(predicted_digit))
        self.conf_label.config(text=f"Confidence: {confidence:.1f}%")

        prob_lines = []
        for d in range(10):
            p = probs[d] * 100.0
            marker = "▶ " if d == predicted_digit else "  "
            prob_lines.append(f"{marker}{d}: {p:5.1f}%")
        self.prob_text.config(text="\n".join(prob_lines))


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerGUI(root)
    root.mainloop()
