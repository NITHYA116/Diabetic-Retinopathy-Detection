# Import the required libraries
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import streamlit as st
import numpy as np

# Load your saved model
model = tf.keras.models.load_model('assets/my_model.h5')

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

    # Set app title and favicon
    st.set_page_config(
        page_title='Diabetic Retinopathy Classification App',
        page_icon=":eye:",
        layout="wide"
    )

    # Define app sidebar
    st.sidebar.image('assets/logo.png', width=250)
    st.sidebar.title('About')
    st.sidebar.info(
        'This app uses a deep learning model to classify diabetic retinopathy.')
    st.sidebar.info(
        'Upload an image of an eye to classify whether the patient has diabetic retinopathy or not.')
    st.sidebar.info(
        'The model was trained on the Kaggle Diabetic Retinopathy dataset, which contains over 35,000 images.')
    st.sidebar.info(
        'To learn more about the dataset, go to https://www.kaggle.com/c/diabetic-retinopathy-detection/overview')

    # Define app header
    header = st.container()
    with header:
        st.title('Diabetic Retinopathy Classification App')
        st.subheader('Upload an image for classification')
        st.markdown('---')

    # Define app main section
    main = st.container()
    with main:
        uploaded_file = st.file_uploader(
            'Choose an image', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            # Display the uploaded image
            image_to_classify = Image.open(uploaded_file)
            st.image(
                image_to_classify,
                caption='Uploaded image',
                use_column_width=True
            )

            # Make predictions on the image
            predictions = predict_image(image_to_classify)
            # Replace with your own class names
            class_names = ['No DR', 'DR']
            top_k = 2  # Display top-2 predictions
            classes = np.argsort(predictions[0])[::-1][:top_k]
            probs = predictions[0][classes]

            # Define prediction section
            prediction = st.container()
            with prediction:
                st.write('## Results')
                st.write('Top-{} Predictions:'.format(top_k))
                for i in range(top_k):
                    st.write(
                        '{}: {:.2%}'.format(class_names[classes[i]], probs[i]),
                        unsafe_allow_html=True
                    )

    # Define app footer
    footer = st.container()
    with footer:
        st.markdown('---')
        st.write('Developed by [Your Name](https://yourwebsite.com)')
        st.write(
            'Code available on [GitHub](https://github.com/yourusername/yourrepository)')


if __name__ == '__main__':
    app()
