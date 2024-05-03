import cv2
import streamlit as st
import numpy as np
import tensorflow as tf

# Set page config to wide layout
st.set_page_config(layout="wide")

# Define function to apply styling
def set_custom_style():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #000000;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: 2px solid #4CAF50;
            border-radius: 4px;
        }
        .stTextInput>div>div>input {
            background-color: #333333;
            color: white;
            border: 2px solid #4CAF50;
            border-radius: 4px;
        }
        .css-145kmo2 {
            border-radius: 8px;
            border: 2px solid #4CAF50;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

set_custom_style()

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

def process_image(original_image, processing_factor):
    st.image(original_image, channels="BGR", caption="Original Image")

    b, g, r = cv2.split(original_image)

    st.image(r, caption="Red Channel")
    st.image(g, caption="Green Channel")
    st.image(b, caption="Blue Channel")

    disease = r - g
    alpha = b

    get_alpha(original_image, alpha)
    st.image(alpha, caption="Alpha Channel")

    for i in range(original_image.shape[0]):
        for j in range(original_image.shape[1]):
            if int(g[i, j]) > processing_factor:
                disease[i, j] = 255

    st.image(disease, caption="Disease Image")
    display_disease_percentage(disease, alpha)


def get_alpha(original_image, alpha):
    for i in range(original_image.shape[0]):
        for j in range(original_image.shape[1]):
            if original_image[i, j, 0] > 200 and original_image[i, j, 1] > 200 and original_image[i, j, 2] > 200:
                alpha[i, j] = 255
            else:
                alpha[i, j] = 0


def display_disease_percentage(disease, alpha):
    count = 0
    res = 0
    for i in range(disease.shape[0]):
        for j in range(disease.shape[1]):
            if alpha[i, j] == 0:
                res += 1
            if disease[i, j] < processing_factor:
                count += 1
    percent = (count / res) * 100
    st.write("Percentage Disease: " + str(round(percent, 2)) + "%")


def get_file():
    return st.file_uploader("Upload an image")

st.title("Plant Disease Detection")
# processing_factor = st.slider("Processing Factor", 0, 255, 150)
processing_factor = 150
file_uploaded = get_file()

if file_uploaded is not None:
    file_bytes = file_uploaded.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    process_image(original_image, processing_factor)
    st.write("Disease Classification:")
    image = tf.keras.preprocessing.image.load_img(file_uploaded, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    predicted_class_index = np.argmax(model.predict(image))
    predicted_class = classes[predicted_class_index]
    st.write("Predicted Class:", predicted_class)
    
else:
    st.write("No File Uploaded!")

