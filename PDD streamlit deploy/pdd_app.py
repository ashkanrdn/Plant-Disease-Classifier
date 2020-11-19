
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import streamlit as st
from PIL import Image, ImageOps


# Predicition function

def plantprediction(testimage, model):

    # built in class names
    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Background_without_leaves',
                   'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                   'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
                   'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato__Early_blight', 'Tomato___Late_blight',
                   'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

    # Image parameters

    img_height = 180
    img_width = 180
    size = (img_width, img_height)

    # load and process image

    img = ImageOps.fit(testimage, size, Image.ANTIALIAS)

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # prediction

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predictionVal = "This plant  most likely has {} problem with a {:.2f} percent confidence.".format(
        class_names[np.argmax(score)], 100 * np.max(score))
    # print (predictionVal)
    return predictionVal


    # Load Model
model = tf.keras.models.load_model('my_model.hdf5')

# StreamLit
st.write("""# Plant Disease Classifier""")

st.write("A simple we app to identify what's wrong with your plant")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=False)
    prediction = plantprediction(image, model)

    st.write(prediction)


# run : streamlit run pdd_app.py
