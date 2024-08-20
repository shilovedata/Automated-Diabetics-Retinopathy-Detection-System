import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the saved model
model = load_model('Diabetics_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Define the classes
class_labels = {
    0: 'No Diabetic Retinopathy (No DR)',
    1: 'Mild Diabetic Retinopathy (Mild DR)',
    2: 'Moderate Diabetic Retinopathy (Moderate DR)',
    3: 'Severe Diabetic Retinopathy (Severe DR)',
    4: 'Proliferative Diabetic Retinopathy (Proliferative DR)'
}

# Streamlit app
st.title('Diabetics Retinopathy Detection')
st.markdown('<style>h1 {color: #ff5733;}</style>', unsafe_allow_html=True)  # Add color to heading

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    image = preprocess_image(uploaded_file)

    # Make predictions
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    # Display the prediction
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(f'Prediction: {class_labels[predicted_class]}, Confidence: {prediction[0][predicted_class]}')
