import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from src.ocr_engine import OCREngine
from src.ai_detector import detect_ai

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

ocr_engine = OCREngine()

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_text_from_image(img_path):
    try:
        text = ocr_engine.extract_text(img_path)
        return text
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

@app.route("/api/detect/text", methods=["POST"])
def detect_text_api():
    data = request.json
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "Text is empty."}), 400
    result = detect_ai(text)
    return jsonify({
        "extracted_text": text,
        "label": result["label"],
        "score": result["score"]
    })

@app.route("/api/detect/image", methods=["POST"])
def detect_image_api():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected image"}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        # Step 1: Extract text using TTCS OCR
        extracted_text = extract_text_from_image(filepath)
        # Step 2: Detect AI from extracted text
        if not extracted_text.strip():
            return jsonify({"error": "No text detected in the image."}), 400
        result = detect_ai(extracted_text)
        return jsonify({
            "extracted_text": extracted_text,
            "label": result["label"],
            "score": result["score"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the file
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    print("Starting Flask API for TTCS on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, threaded=False, use_reloader=False)
