import streamlit as st

st.title("Deep Learning")
model_training_state = st.text("Model is being trained. This can take several minutes.")

from deep_learning_fashion_mnist_dataset_epitech import *

model_training_state.text("Model successfully trained.")


def evaluate_images(*args):
    print("In callback")


uploader = st.file_uploader(
    "Upload files to pass to the model",
    type=["png", "jpg"],
    accept_multiple_files=True,
    key=None,
    help=None,
    on_change=evaluate_images,
    args=None,
    kwargs=None,
)
