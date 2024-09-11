import global_variables as enum
import matplotlib.pyplot as plt
import os

from testing import prepare_image, predict_image
from sklearn import metrics

# Building the arrays to build the confusion matrix
actual_result = []
predict_result = []

base_dir = './data_set/working/test/'

for class_name in os.listdir(base_dir):
  class_dir = os.path.join(base_dir, class_name)
  
  if os.path.isdir(class_dir):
    for image_name in os.listdir(class_dir):
      image_path = os.path.join(class_dir, image_name)

      actual_result.append(class_name)

      pred_label, _ = predict_image(image_path)
      predict_result.append(enum.GlobalVariables.CLASS_IMAGES.value[pred_label])

# Showing the confusion matrix
fig, ax = plt.subplots(figsize=(12, 10))
confusion_matrix = metrics.confusion_matrix(actual_result, predict_result)
cm_display = metrics.ConfusionMatrixDisplay(
  confusion_matrix = confusion_matrix,
  display_labels = enum.GlobalVariables.CLASS_IMAGES.value
)
cm_display.plot(ax=ax)
plt.xticks(rotation=45, ha='right')
plt.show()