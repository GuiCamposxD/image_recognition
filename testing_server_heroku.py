from flask import Flask, request, jsonify
import os
from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

app = Flask(__name__)

CLASS_IMAGES = [
    "apple", "banana", "beetroot", "bell pepper", "cabbage", "capsicum", "carrot",
    "cauliflower", "chilli pepper", "corn", "cucumber", "eggplant", "garlic",
    "ginger", "grapes", "jalepeno", "kiwi", "lemon", "lettuce", "mango"
  ]

# Load the model
model = tf.keras.models.load_model('./model/model5.keras')

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

# API endpoint to handle image uploads
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Process the image and get the prediction
        img_stream = file.stream  # This is an in-memory stream of the file
        pred_class, confidence = predict_image(img_stream)

        # Get the predicted fruit or vegetable class name
        predicted_label = CLASS_IMAGES[pred_class]
        
        return jsonify({
            "predicted_label": predicted_label,
            "confidence": "{:.2f}".format(confidence * 100)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5353))  # Use Heroku's dynamic port
    app.run(debug=True, host='0.0.0.0', port=port)

