
from flask import Flask, request, jsonify
from flask_cors import CORS

from backend.prediction import (
    MODEL_REGISTRY,
    predict_xray
)


app = Flask(__name__)

CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "FedXray backend is running."
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "healthy",
        "service": "FedXray API"
    })


@app.route("/api/models", methods=["GET"])
def get_models():

    models = []

    for model_id, model_info in MODEL_REGISTRY.items():

        configurations = []

        for config_id, config_info in model_info["configurations"].items():

            if model_id == "centralized" and config_id == "default":
                continue

            configurations.append({
                "id": config_id,
                "label": config_info["label"],
                "available": True
            })

        models.append({
            "id": model_id,
            "name": model_info["name"],
            "available": True,
            "configurations": configurations
        })

    return jsonify({
        "success": True,
        "models": models,
        "threshold_modes": [
            {
                "id": "default",
                "label": "Default Threshold (0.5)"
            },
            {
                "id": "tuned",
                "label": "Tuned Threshold (Validation Optimized)"
            }
        ]
    })


@app.route("/api/predict", methods=["POST"])
def predict():

    image_file = request.files.get("image")

    if image_file is None:
        return jsonify({
            "success": False,
            "detail": "Please upload an image using the 'image' field."
        }), 400

    if not image_file.filename:
        return jsonify({
            "success": False,
            "detail": "No image file was selected."
        }), 400

    allowed_extensions = {"png", "jpg", "jpeg"}

    extension = image_file.filename.rsplit(".", 1)[-1].lower()

    if "." not in image_file.filename or extension not in allowed_extensions:
        return jsonify({
            "success": False,
            "detail": "Only PNG, JPG and JPEG images are supported."
        }), 400

    model_id = request.form.get("model_id", "centralized")
    config_id = request.form.get("config_id", "standard")
    threshold_mode = request.form.get("threshold_mode", "default")

    try:

        result = predict_xray(
            image_file=image_file.stream,
            model_id=model_id,
            config_id=config_id,
            threshold_mode=threshold_mode
        )

        return jsonify(result)

    except (ValueError, FileNotFoundError) as error:

        return jsonify({
            "success": False,
            "detail": str(error)
        }), 400

    except Exception:

        app.logger.exception("Prediction failed")

        return jsonify({
            "success": False,
            "detail": "Prediction failed. Check the backend terminal."
        }), 500


@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "detail": "Image is too large. Maximum size is 15 MB."
    }), 413


if __name__ == "__main__":

    print("Starting FedXray backend...")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=False
    )
