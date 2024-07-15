import cv2
import numpy as np
import os
from io import BytesIO
import IPython
import requests
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageTk
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from keras.utils import load_img, img_to_array
import tkinter as tk

STABILITY_KEY = 'sk-EZvd1qLoQfO8kqH00ILWcSbcvKCD4GGbpgJuCcdR5s47o7M1'

# Global variables for masking
rectangles = []

def select_rectangle(event):
    global start_x, start_y, rect_id
    start_x = event.x
    start_y = event.y
    rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline='red')

def update_rectangle(event):
    canvas.coords(rect_id, start_x, start_y, event.x, event.y)

def save_rectangle(event):
    end_x = event.x
    end_y = event.y
    rectangles.append((start_x, start_y, end_x, end_y))
    canvas.unbind("<B1-Motion>")
    canvas.unbind("<ButtonRelease-1>")
    canvas.bind("<Button-1>", select_rectangle)
    canvas.bind("<B1-Motion>", update_rectangle)
    canvas.bind("<ButtonRelease-1>", save_rectangle)

def create_masked_image(image_path):
    global canvas, img_tk

    root = tk.Tk()
    root.title("Select Regions to Mask")

    img = Image.open(image_path)
    img_tk = ImageTk.PhotoImage(img)
    canvas = tk.Canvas(root, width=img.width, height=img.height)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    canvas.bind("<Button-1>", select_rectangle)
    canvas.bind("<B1-Motion>", update_rectangle)
    canvas.bind("<ButtonRelease-1>", save_rectangle)

    root.mainloop()

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    for rect in rectangles:
        start_x, start_y, end_x, end_y = rect
        start_x, start_y = max(start_x, 0), max(start_y, 0)
        end_x, end_y = min(end_x, img.shape[1]), min(end_y, img.shape[0])
        img[start_y:end_y, start_x:end_x, 3] = 0

    output_path = 'output_image.png'
    cv2.imwrite(output_path, img)
    print(f"Image saved to {output_path}")

    return output_path

def send_generation_request(host, params):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files) == 0:
        files["none"] = ''

    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return response

def edit_image_with_stability(image_path, prompt, output_format='jpeg', strength=0.75, seed=0):
    host = f"https://api.stability.ai/v2beta/stable-image/edit/inpaint"

    params = {
        "image" : image_path,
        "prompt" : prompt,
        "strength" : strength,
        "seed" : seed,
        "output_format": output_format,
        "mode" : "image-to-image",
        "model" : "sd3-medium"
    }

    response = send_generation_request(host, params)

    output_image = response.content
    finish_reason = response.headers.get("finish-reason")
    seed = response.headers.get("seed")

    if finish_reason == 'CONTENT_FILTERED':
        raise Warning("Generation failed NSFW classifier")

    filename, _ = os.path.splitext(os.path.basename(image_path))
    edited = f"generated_{seed}.{output_format}"
    with open(edited, "wb") as f:
        f.write(output_image)
    print(f"Saved image {edited}")

    return edited

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
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

    reference_features = extract_features(model, reference_img_path)

    image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
    similarities = []

    for img_path in image_paths:
        features = extract_features(model, img_path)
        similarity = cosine_similarity([reference_features], [features])[0][0]
        similarities.append((img_path, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    similar_images = [img[0] for img in similarities[:top_n]]

    return similar_images

# Main flow
if __name__ == "__main__":
    image_path = 'white shirt.jpg'  # Replace with your image path
    prompt = "Style outfit with denim jackets"
    folder_path = 'fashion'  # Replace with the folder containing images

    masked_image_path = create_masked_image(image_path)
    edited_image_path = edit_image_with_stability(masked_image_path, prompt)
    similar_images = find_similar_images(edited_image_path, folder_path)

    print("Original image:")
    IPython.display.display(Image.open(image_path))
    print("Masked image:")
    IPython.display.display(Image.open(masked_image_path))
    print("Edited image:")
    IPython.display.display(Image.open(edited_image_path))

    print("Similar images:")
    for img_path in similar_images:
        print(img_path)
        IPython.display.display(Image.open(img_path))