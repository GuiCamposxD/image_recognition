from enum import Enum
from tensorflow.keras.utils import image_dataset_from_directory # type: ignore

class GlobalVariables(Enum):
  BATCH_SIZE = 64
  EPOCHS = 100
  IMG_SIZE = (128, 128)
  INPUT_SHAPE = (128, 128, 3)
  NUMBER_CLASSES = 36
  TEST_PATH = './data_set/working/test'
  TRAIN_PATH = './data_set/working/train'
  VAL_PATH = './data_set/working/validation'
  CLASS_IMAGES = image_dataset_from_directory(TEST_PATH).class_names

for i in range(0, 20):
  print(GlobalVariables.CLASS_IMAGES.value[i])