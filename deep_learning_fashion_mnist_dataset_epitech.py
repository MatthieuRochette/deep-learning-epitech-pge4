# -*- coding: utf-8 -*-
"""deep-learning-fashion-mnist-dataset-epitech.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ie1sdvXXUqbQq8If10mQAeuJU7_sScyW
"""

# Commented out IPython magic to ensure Python compatibility.
# Load the TensorBoard notebook extension
# %load_ext tensorboard
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sys import version_info
import datetime
import os
print(version_info)

# Commented out IPython magic to ensure Python compatibility.
# Clear any logs from previous runs
# %rm -rf ./logs/

# Load the data set
def load_dataset():
  fashion_mnist = tf.keras.datasets.fashion_mnist

  return fashion_mnist.load_data()


(train_images, train_labels), (test_images, test_labels) = load_dataset()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# turn the gray scale into float entrer 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Creation of model 

def create_model():
  model = keras.models.Sequential()

  model.add(keras.layers.Input((28,28,1)))

  # We add one layer for convolution with 32 filters
  model.add(keras.layers.Conv2D(320, (3,3),  activation='relu')) #32

  # We add pooling layer
  model.add(keras.layers.MaxPooling2D((2,2)))

  # We add Dropout layer for reset the state for neral
  model.add(keras.layers.Dropout(0.2))

  model.add(keras.layers.Flatten())
  # We add one layer of normal network layer
  model.add(keras.layers.Dense(640, activation='relu')) #128
  model.add(keras.layers.Dropout(0.5))

  # And for finish we add a final layer of final clasification
  model.add(keras.layers.Dense(100, activation='softmax')) #10
  return model

if __name__ == "__main__":
  MODEL_WAS_LOADED = False
  if os.path.isdir("./saved_model"):
    model = keras.models.load_model(path)
    MODEL_WAS_LOADED = True
  else:
    model = create_model()

if __name__ == "__main__":
  model.summary()

# compile the model with adam optimizer for reduse the lost teste
def compile_model_with_adam(model):
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__" and not MODEL_WAS_LOADED:
    model = compile_model_with_adam(model)

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

checkpoint_path = "training_saves/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

def fit_model(model, epochs: int = 20):
  # train the model with 20 epochs with the train dataset
  model.fit(train_images, train_labels, epochs=20, callbacks=[tensorboard_callback, cp_callback])
  return  model

if __name__ == "__main__" and not MODEL_WAS_LOADED:
  if os.path.isdir(checkpoint_dir):  # load saved weights if available
    model.load_weights(checkpoint_path)
    
    # USE A FIT HERE IF YOU WANT TO COMPLETE AN INTERRUPTED TRAINING
    # epoch parameter will differ each time, so change it yourself:
    # remaining_epochs_to_complete = X
    # model = fit.model(model, epochs=remaining_epochs_to_complete)
  else:
    model = fit_model(model)

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir logs

if __name__ == "__main__":
  score = model.evaluate(test_images, test_labels, verbose=0)

  print(f'Test loss     : {score[0]:4.4f}')
  print(f'Test accuracy : {score[1]:4.4f}')

if __name__ == "__main__" and not MODEL_WAS_LOADED:
  model.save("./saved_model")

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

if __name__ == "__main__":
  probability_model = tf.keras.Sequential([model, 
                                           tf.keras.layers.Softmax()])

if __name__ == "__main__":
  predictions = probability_model.predict(test_images)

# Plot the first X test images, their predicted labels, and the true labels.
  # Color correct predictions in blue and incorrect predictions in red.
  num_rows = 10
  num_cols = 3
  num_images = num_rows*num_cols
  plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
  plt.tight_layout()
  plt.show()