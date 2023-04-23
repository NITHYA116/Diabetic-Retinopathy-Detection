# Import the required libraries
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st
import numpy as np

# Load your saved model
model = tf.keras.models.load_model('my_model.h5')

# Define the prediction function


def predict_image(image):
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Rescale pixel values to [0, 1]
    img_batch = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_batch)
    return preds

# Define the Streamlit app


def app():
    st.set_page_config(page_title='Image Classification App')
    st.title('Image Classification App')
    st.write('Upload an image for classification')

    # Create a file uploader
    uploaded_file = st.file_uploader(
        'Choose an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        image_to_classify = image.load_img(uploaded_file)
        st.image(image_to_classify, caption='Uploaded image',
                 use_column_width=True)

        # Make predictions on the image
        predictions = predict_image(image_to_classify)
        # Replace with your own class names
        class_names = ['No_DR', 'DR']
        top_k = 2  # Display top-3 predictions
        classes = np.argsort(predictions[0])[::-1][:top_k]
        probs = predictions[0][classes]

        st.write('Top-{} Predictions:'.format(top_k))
        for i in range(top_k):
            st.write('{}: {:.2%}'.format(class_names[classes[i]], probs[i]))


if __name__ == '__main__':
    app()
