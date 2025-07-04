"""
HematoVision – Flask backend
Upload a blood‑cell image → predict Neutrophil / Monocyte / Lymphocyte / Eosinophil
"""

import os
import pathlib
import uuid
import numpy as np
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ────────────── Configuration ──────────────
BASE_DIR   = pathlib.Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_PATH = BASE_DIR / "model" / "model.h5"
CLASS_LABELS = ["neutrophil", "monocyte", "lymphocyte", "eosinophil"]
IMG_SIZE = (224, 224)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ────────────── Flask app ──────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.secret_key = "dev"
model = load_model(MODEL_PATH)

# ────────────── Routes ──────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # ---------- 1. validate file ----------
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    # ---------- 2. save upload ----------
    filename  = secure_filename(file.filename)
    # add a uuid so re‑uploads don’t clash
    filename  = f"{uuid.uuid4().hex}_{filename}"
    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    # ---------- 3. preprocess ----------
    img = load_img(save_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)

    # ---------- 4. inference ----------
    probs = model.predict(arr)[0]
    pred_class = CLASS_LABELS[int(np.argmax(probs))]
    confidence = round(float(np.max(probs)) * 100, 2)

    # ---------- 5. show result ----------
    image_url = url_for("static", filename=f"uploads/{filename}")
    return render_template(
        "result.html",
        image_path=image_url,
        prediction=pred_class,
        confidence=f"{confidence}%"
    )

# ────────────── Run in dev mode ──────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # set debug=False in prod
