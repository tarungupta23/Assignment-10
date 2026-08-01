"""
Task 3: API Development

REST API that loads the trained model, accepts patient clinical
parameters as JSON, and returns a heart-disease-risk prediction as JSON.

"""

import pickle
import os
import warnings
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings("ignore", message="X does not have valid feature names")

app = Flask(__name__)

MODEL_PATH = "model.pkl"
META_PATH = "model_meta.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

if os.path.exists(META_PATH):
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    FEATURES = meta["features"]
else:
    FEATURES = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    ]


@app.route("/", methods=["GET"])
def home():
    """Simple optional web form for manual testing."""
    return render_template("index.html", features=FEATURES)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing required fields: {missing}"
            }), 400

        input_row = [[float(data[f]) for f in FEATURES]]
        prediction = model.predict(input_row)[0]

        result = "Heart Disease Detected" if int(prediction) == 1 else "No Heart Disease Detected"

        response = {"prediction": result}

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_row)[0]
            response["confidence"] = round(float(max(proba)), 4)

        return jsonify(response), 200

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
