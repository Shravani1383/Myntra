import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from keras.utils import load_img, img_to_array
from PIL import Image

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def extract_features(model, img_path):
    img_array = load_and_preprocess_image(img_path)
    features = model.predict(img_array)
    return features.flatten()

def find_similar_images(reference_img_path, folder_path, top_n=5):
    # Load pre-trained VGG16 model + higher level layers
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    # Extract features from the reference image
    reference_features = extract_features(model, reference_img_path)

    # Extract features from all images in the folder
    image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
    similarities = []

    for img_path in image_paths:
        features = extract_features(model, img_path)
        similarity = cosine_similarity([reference_features], [features])[0][0]
        similarities.append((img_path, similarity))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Get the top_n similar images
    similar_images = [img[0] for img in similarities[:top_n]]

    return similar_images

# Example usage
reference_img_path = 'white shirt.jpg'
folder_path = 'fashion'
similar_images = find_similar_images(reference_img_path, folder_path)

print("Similar images:")
for img_path in similar_images:
    print(img_path)
    # Display the images (optional)
    img = Image.open(img_path)
    img.show()
