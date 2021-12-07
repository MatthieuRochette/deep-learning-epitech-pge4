import os
import io
from PIL import Image, ImageOps

import numpy as np
import streamlit as st
from tensorflow import keras

from deep_learning_fashion_mnist_dataset_epitech import *

# CONSTANTS
# =============================================================================
DEFAULT_MODEL_SAVE_PATH = "./saved_model"

# streamlit initialisation
# =============================================================================
st.title("Deep Learning")
model_training_state = st.text(
    """Deep Learning model is being trained or the page is being reloaded.
In the first case, this might take a long time. Sorry for the inconvenience."""
)

# model initialisation
# =============================================================================
def try_load_model(path: str = DEFAULT_MODEL_SAVE_PATH):
    if os.path.isdir(path):
        model = keras.models.load_model(path)
        return model
    return None


def load_or_create_model(path: str = DEFAULT_MODEL_SAVE_PATH):
    model = try_load_model()
    if model is None:
        model = create_model()
        model = compile_model_with_adam(model)
        model = fit_model(model)
        model.save(path)
    return model


model = load_or_create_model()
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
model_training_state.text(
    """Deep learning model is ready.
You can test it by providing your own images hereunder."""
)

# streamlit file input widget
# =============================================================================
uploaded = st.file_uploader(
    "Upload files to pass to the model",
    type=["png", "jpg"],
    accept_multiple_files=True,
    key=None,
    help=None,
    args=None,
    kwargs=None,
)

# streamlit button launch prediction
# =============================================================================
if uploaded:
    converted_images = []
    images_names = []
    images = []
    for file in uploaded:
        image = Image.open(io.BytesIO(file.getvalue()))
        images.append(image)
        if image.size[0] != 28 or image.size[1] != 28:
            image = image.resize((28, 28))
        image_grayscale = ImageOps.grayscale(image)

        images_names.append(file.name)

        pixels = list(image_grayscale.getdata())
        pixels_matrix = [[*pixels[i : i + 28]] for i in range(0, 28)]
        pixels_np_arr = np.array([np.array(pix_line) for pix_line in pixels_matrix])

        converted_images.append(pixels_np_arr)

    converted_images_array = np.array(converted_images)
    predictions = probability_model.predict(converted_images_array)

    print(len(predictions))
    for i in range(len(images_names)):
        img_predictions_array = predictions[i]
        predicted_label = np.argmax(img_predictions_array)
        predicted_class = class_names[predicted_label]

        with st.expander(images_names[i], expanded=True):
            st.image(images[i])
            st.caption(predicted_class)
            print(images_names[i], predictions[i])
