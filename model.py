import global_variables as enum

from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Rescaling, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Building Model
def create_model():
  model = Sequential()

  model.add(Conv2D(128, (3, 3), activation='relu', padding='same')),
  model.add(BatchNormalization()),
  model.add(MaxPooling2D((2, 2))),

  model.add(Conv2D(128, (3, 3), activation='relu', padding='same')),
  model.add(BatchNormalization()),
  model.add(MaxPooling2D((2, 2))),

  model.add(Conv2D(64, (3, 3), activation='relu', padding='same')),
  model.add(BatchNormalization()),
  model.add(MaxPooling2D((2, 2))),

  model.add(Conv2D(128, (3, 3), activation='relu', padding='same')),
  model.add(BatchNormalization()),
  model.add(MaxPooling2D((2, 2))),
  
  model.add(Conv2D(32, (3, 3), activation='relu', padding='same')),
  model.add(BatchNormalization()),
  model.add(MaxPooling2D((2, 2))),

  model.add(Flatten()),

  model.add(Dense(512, activation='relu')),
  model.add(BatchNormalization()),

  model.add(Dense(64, activation='relu')),
  model.add(BatchNormalization()),
  model.add(Dropout(0.3)),
  
  model.add(Dense(units = enum.GlobalVariables.NUMBER_CLASSES.value, activation='softmax'))

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  return model