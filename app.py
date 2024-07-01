import cv2
import numpy as np
import streamlit as st
import pickle
import tensorflow as tf 

def predict_mask(image_path, model):

 try:
    image = cv2.imread(image_path)

    if image is None:
      raise ValueError(f"Error: Could not read image from '{image_path}'. Please check the path or file format.")
    image_resized = cv2.resize(image, (128, 128))
    image_scaled = image_resized.astype('float32') / 255.0
    image_reshaped = np.expand_dims(image_scaled, axis=0)

    prediction = model.predict(image_reshaped)
    pred_label = np.argmax(prediction)

    return pred_label

 except Exception as e:
    st.error(f"Error: {e}")  
    return None  

model = tf.keras.models.load_model('model.h5')


st.title("Face Mask Detection")
uploaded_file = st.file_uploader("Choose an image:", type="jpg")

if st.button('Show Results'):
    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        image = cv2.imdecode(np.fromstring(image_bytes, np.uint8), 1)
        st.image(image, channels="BGR")
        image_resized = cv2.resize(image, (128, 128))
        image_scaled = image_resized.astype('float32') / 255.0
        image_reshaped = np.reshape(image_scaled, [1, 128, 128, 3])
        prediction = model.predict(image_reshaped)
        if prediction[0] <= 0.5:
            st.write('The person in the image is wearing a mask')

        else:
            st.write('The person in the image is not wearing a mask')

