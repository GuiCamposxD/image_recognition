import global_variables as enum
import pickle

from model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Pre-processing data for CNN
BATCH_SIZE = 20

# Create an instance of the ImageDataGenerator for the test set (usually without augmentation)
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create the generators
train = train_datagen.flow_from_directory(
  enum.GlobalVariables.TRAIN_PATH.value,
  seed=777,
  target_size=enum.GlobalVariables.IMG_SIZE.value,
  batch_size=enum.GlobalVariables.BATCH_SIZE.value,
  class_mode='categorical'
)

test = test_datagen.flow_from_directory(
  enum.GlobalVariables.TEST_PATH.value,
  seed=777,
  target_size=enum.GlobalVariables.IMG_SIZE.value,
  batch_size=enum.GlobalVariables.BATCH_SIZE.value,
  class_mode='categorical'
)

val = test_datagen.flow_from_directory(
  enum.GlobalVariables.VAL_PATH.value,
  seed=777,
  target_size=enum.GlobalVariables.IMG_SIZE.value,
  batch_size=enum.GlobalVariables.BATCH_SIZE.value,
  class_mode='categorical'
)

# Define callbacks
checkpoint_callback = ModelCheckpoint(
  './model/model3.keras',
  monitor='val_accuracy',
  save_best_only=True,
  mode='max',
  verbose=1
)

# Traning the model
model = create_model()

history = model.fit(
  train,
  epochs=enum.GlobalVariables.EPOCHS.value,
  batch_size=enum.GlobalVariables.BATCH_SIZE.value,
  validation_data=val,
  verbose=2,
  callbacks=[checkpoint_callback]
)

with open('training_history.pkl', 'wb') as f:
  pickle.dump(history.history, f)

model.save('model/model3_save.keras')
