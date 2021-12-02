import streamlit as st

st.title("Deep Learning")
model_training_state = st.text("Model is being trained. This can take several minutes.")

import deep_learning_fashion_mnist_dataset_epitech

model_training_state.text("Model successfully trained.")
