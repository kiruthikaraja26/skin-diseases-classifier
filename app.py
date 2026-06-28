from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from tensorflow.keras.models import load_model

from preprocess import preprocess_image
from database import init_db, insert_log
from disease_info import disease_data

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "../model/skin_model.h5"
CLASSES = ['melanoma', 'eczema', 'acne', 'psoriasis']

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model(MODEL_PATH)

# Init DB
init_db()


@app.route("/")
def home():
    return {"message": "Smart Skin Disease API Running 🚀"}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Preprocess
        img = preprocess_image(filepath)

        # Predict
        preds = model.predict(img)[0]
        class_index = np.argmax(preds)
        label = CLASSES[class_index]
        confidence = float(preds[class_index])

        # Get disease info
        info = disease_data.get(label, {})

        # Severity logic
        severity = info.get("severity", "low")

        if severity == "high":
            warning = "⚠️ This may be a serious condition. Consult a dermatologist immediately."
        elif severity == "medium":
            warning = "⚠️ This condition should be monitored. Consider consulting a doctor."
        else:
            warning = "✅ This appears mild. Follow basic care."

        # Save to DB
        insert_log(file.filename, label, confidence)

        return jsonify({
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "description": info.get("description", "No info available"),
            "solutions": info.get("solution", []),
            "severity": severity,
            "advice": warning
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)