import tensorflow as tf
import numpy as np

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

def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

def predict_class(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = classes[predicted_class_index]
    return predicted_class

image_path = 'soyabean.jpg'
predicted_class = predict_class(image_path)
print("Predicted Class:", predicted_class)
