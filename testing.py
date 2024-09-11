import global_variables as enum
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load model
model = tf.keras.models.load_model('./model/model5.keras')

# Define function to predict an image
def prepare_image(img_path):
  img = load_img(img_path, target_size=enum.GlobalVariables.IMG_SIZE.value)
  img_array = img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array = img_array / 255.0

  return img_array

def predict_image(img_path):
  img_array = prepare_image(img_path)
  pred_prob = model.predict(img_array)
  pred_class = np.argmax(pred_prob, axis=1)[0]
  
  return (pred_class, pred_prob)

pred_label, score = predict_image('./maca.jpg')

print('Veg/Fruit in image is {} with accuracy of {:0.2f}'.format(enum.GlobalVariables.CLASS_IMAGES.value[pred_label], np.max(score)*100))
