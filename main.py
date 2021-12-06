import os

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
    for image in uploaded:
        print(image)
    # predictions = probability_model.predict(images_from_uploader)
