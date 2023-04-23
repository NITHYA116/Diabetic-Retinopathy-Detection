import streamlit as st
import tensorflow.keras as keras
import numpy as np
from PIL import Image

# Load the model
model = keras.models.load_model('my_model.h5')

# Define the input shape of the model
input_shape = (224, 224, 3)

# Define a function to preprocess the image


def preprocess_image(image):
    # Resize the image to the input shape of the model
    image = image.resize(input_shape[:2])

    # Convert the image to a numpy array
    img_array = np.array(image)

    # Normalize the pixel values
    img_array = img_array / 255.0

    # Expand the dimensions of the image to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Define a function to make predictions with the model


def predict(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make a prediction with the model
    prediction = model.predict(processed_image)

    # Return the predicted class
    return prediction


# Define the layout of the app
st.title('Image Classification App')
file = st.file_uploader('Upload an image')

if file is not None:
    # Read the image
    image = Image.open(file)

    # Make a prediction with the model
    prediction = predict(image)

    # Get the predicted class label
    class_index = np.argmax(prediction)
    # Replace with your own class labels
    labels = ['class_0', 'class_1', 'class_2']
    predicted_class = labels[class_index]

    # Display the image and prediction
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write('Prediction:', predicted_class, prediction)
