import os
import pickle
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
print("[INFO] Loading model...")
model = load_model('plant_disease_classification_model.h5')

# Load the label transform
print("[INFO] Loading label transform...")
filename = 'plant_disease_label_transform.pkl'
image_labels = pickle.load(open(filename, 'rb'))

# Dimension of resized image
DEFAULT_IMAGE_SIZE = tuple((256, 256))

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

def predict_disease(image_path):
    image_array = convert_image_to_array(image_path)
    np_image = np.array(image_array, dtype=np.float16) / 225.0
    np_image = np.expand_dims(np_image, 0)
    # plt.imshow(plt.imread(image_path))
    raw_predictions = model.predict(np_image)
    predicted_class_index = np.argmax(raw_predictions)
    predicted_class = image_labels.classes_[predicted_class_index]
    print(predicted_class)

# Example usage:
# val_dir = "path/to/your/validation/directory"
image_path = 'photo_4.JPG'
predict_disease(image_path)
