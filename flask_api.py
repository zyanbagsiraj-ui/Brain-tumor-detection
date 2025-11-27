# flask_api.py - FULL FINAL CODE (100% WORKING + CORRECT CLASS ORDER)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from load_member2_model import load_member2_model
from validation import validate_image
import os

app = Flask(__name__)
CORS(app)

# Load model once at startup
print("Loading the 92%+ accuracy model...")
model = load_member2_model()
print("Model loaded and ready!")

# CORRECT CLASS ORDER (matches your dataset folders)
classes = ['notumor', 'glioma', 'meningioma', 'pituitary']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    temp_path = "temp_upload.jpg"
    file.save(temp_path)

    try:
        # Validation (Member 3)
        is_valid, message = validate_image(temp_path)
        if not is_valid:
            return jsonify({"error": message}), 400

        # Preprocess image
        img = load_img(temp_path, target_size=(224, 224), color_mode="rgb")
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array, verbose=0)[0]
        confidence = float(np.max(prediction) * 100)
        predicted_class = classes[np.argmax(prediction)]

        # All probabilities
        probabilities = {classes[i]: round(float(prediction[i] * 100), 2) 
                        for i in range(len(classes))}

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)