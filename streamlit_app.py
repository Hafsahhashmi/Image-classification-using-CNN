import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('mnist_cnn_model.h5')

# Define the Streamlit app
st.title("MNIST Digit Classification")

# File uploader for user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Convert the uploaded image to an array and preprocess it
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    
    # Predict the class of the image
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)
    
    st.image(uploaded_file, caption=f'Uploaded Image', use_column_width=True)
    st.write(f"Predicted Digit: {predicted_class[0]}")
