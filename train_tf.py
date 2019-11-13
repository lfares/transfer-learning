from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

AUTOTUNE = tf.data.experimental.AUTOTUNE

# import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pathlib

import os

# Auxiliar functions
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
      plt.show()

class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

PATH = os.getcwd()
train_dir = pathlib.Path(PATH+'/cats_and_dogs_filtered/train')
validate_dir = pathlib.Path(PATH+'/cats_and_dogs_filtered/validation')

CLASS_NAMES = np.array([item.name for item in train_dir.glob('*')])

print(CLASS_NAMES)

train_image_count = len(list(train_dir.glob('*/*.jpg')))
validate_image_count = len(list(validate_dir.glob('*/*.jpg')))
# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 32
IMG_HEIGHT = 160
IMG_WIDTH = 160

train_data_gen = image_generator.flow_from_directory(directory=str(train_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES),
                                                     class_mode='categorical')

validate_data_gen = image_generator.flow_from_directory(directory=str(validate_dir),
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        classes = list(CLASS_NAMES),
                                                        class_mode='categorical')

image_batch, label_batch = next(train_data_gen)

# Test pre-processed data
# show_batch(image_batch, label_batch)

# Download headless model
model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(160,160,3), include_top=False, weights='imagenet')
# feature_extractor_layer.trainable = False

# Attach classification head
# model = tf.keras.Sequential([
#   feature_extractor_layer,
#   #tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid')
#   layers.Dense(train_data_gen.num_classes, activation='softmax')
# ])

model.summary()

predictions = model(image_batch)
print(predictions.shape)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

steps_per_epoch = np.ceil(train_data_gen.samples/train_data_gen.batch_size)
validation_steps = np.ceil(validate_data_gen.samples/validate_data_gen.batch_size)

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='auto')
batch_stats_callback = CollectBatchStats()
history = model.fit_generator(train_data_gen,
                              validation_data= validate_data_gen,
                              epochs=2,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps,
                              callbacks = [earlystop])

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cats_and_dogs_trained_model.h5'
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)