from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import global_variables as enum

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('./model/model3.keras')

# Prepare the image before making predictions
def prepare_image(img):
    img = load_img(img, target_size=enum.GlobalVariables.IMG_SIZE.value)
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
        img_path = file.stream  # file is in-memory stream
        pred_class, confidence = predict_image(img_path)

        # Get the predicted fruit or vegetable class name
        predicted_label = enum.GlobalVariables.CLASS_IMAGES.value[pred_class]
        
        return jsonify({
            "predicted_label": predicted_label,
            "confidence": "{:.2f}".format(confidence * 100)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
