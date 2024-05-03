import cv2
import streamlit as st
import numpy as np
import tensorflow as tf

path = 'plantvillage_dataset/color'

train_ds, test_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    image_size=(224, 224),
    batch_size=32,
    seed=123,
    validation_split=.2,
    subset='both'
)
classes = train_ds.class_names

model = tf.keras.models.load_model('Plant_disease.h5')

def process_image(original_image, disease_image):
    st.image(original_image, channels="BGR", caption="Original Image")
    st.image(disease_image, channels="BGR", caption="Image with Disease")

    # Disease classification
    st.write("Disease Classification:")
    image = tf.keras.preprocessing.image.load_img(file_uploaded, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    predicted_class_index = np.argmax(model.predict(image))
    predicted_class = classes[predicted_class_index]
    st.write("Predicted Class:", predicted_class)


def get_file():
    return st.file_uploader("Upload an image")

st.title("PlantSoul")
file_uploaded = get_file()

if file_uploaded is not None:
    file_bytes = file_uploaded.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the image to detect disease
    disease_image = original_image.copy()
    # Add your disease detection logic here
    # For example, let's simulate a disease by adding noise to the image
    noise = np.random.normal(0, 30, original_image.shape)
    disease_image = np.clip(disease_image + noise, 0, 255).astype(np.uint8)

    process_image(original_image, disease_image)
    
else:
    st.write("No File Uploaded!")
