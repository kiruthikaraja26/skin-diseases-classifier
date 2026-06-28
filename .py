import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the model you trained in Colab
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('skin_disease_model.h5')

model = load_my_model()
classes = ['Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Vascular Lesions']

st.set_page_config(page_title="Skin-AI Diagnostic", layout="centered")
st.title("🩺 Skin Disease Classifier")
st.write("Upload a dermoscopic image for real-time analysis[cite: 11].")

file = st.file_uploader("Please upload an image (jpg, png)", type=["jpg", "png"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Pre-processing to match training data
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    img_array = np.asarray(image) / 255.0
    img_reshape = img_array[np.newaxis, ...]

    if st.button("Predict"):
        prediction = model.predict(img_reshape)
        idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        st.subheader(f"Result: {classes[idx]}")
        st.progress(int(confidence))
        st.write(f"Confidence Score: {confidence:.2f}% [cite: 30]")