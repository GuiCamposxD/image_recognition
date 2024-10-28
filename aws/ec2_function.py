import os
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
import numpy as np
import tensorflow as tf

app = Flask(__name__)

CLASS_IMAGES = [
    "apple", "banana", "beetroot", "bell pepper", "cabbage", "capsicum", "carrot",
    "cauliflower", "chilli pepper", "corn", "cucumber", "eggplant", "garlic",
    "ginger", "grapes", "jalepeno", "kiwi", "lemon", "lettuce", "mango"
  ]

# Path to save temporary images
TEMP_IMAGE_PATH = '/mnt/temp-images'

# Load the model
model = tf.keras.models.load_model('./model5.keras')
print("Model loaded successfully.")  # Debug statement to confirm model loading

# Prepare the image before making predictions
def prepare_image(img_stream):
    # Convert the image stream (SpooledTemporaryFile) to BytesIO
    img = BytesIO(img_stream.read())
    
    # Use load_img with the BytesIO stream
    img = load_img(img, target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Predict fruit or vegetable in the image
def predict_image(img):
    img_array = prepare_image(img)
    pred_prob = model.predict(img_array)
    pred_class = np.argmax(pred_prob, axis=1)[0]
    return pred_class, np.max(pred_prob)

@app.route("/predict", methods=["POST"])
def predict():
    print("Prediction request received.")  # Debug statement
    if 'file' not in request.files:
        print("No file found in the request.")  # Debug statement
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        print("File part found but no file selected.")  # Debug statement
        return jsonify({"error": "No selected file"}), 400

    try:
        # Process the image and get the prediction
        print("Image prepared, making prediction...")  # Debug statement
        img_stream = file.stream  # This is an in-memory stream of the file
        pred_class, confidence = predict_image(img_stream)

        # Get the predicted fruit or vegetable class name
        predicted_label = CLASS_IMAGES[pred_class]

        print(f"Prediction successful: Class={pred_class}, Confidence={confidence}")  # Debug statement
        return jsonify({
            "predicted_label": predicted_label,
            "confidence": "{:.2f}".format(confidence * 100)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")  # Debug statement
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
