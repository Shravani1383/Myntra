import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import requests
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.utils import load_img, img_to_array
from sklearn.metrics.pairwise import cosine_similarity

STABILITY_KEY = 'sk-XX7WgxA5tsMp500j0z5J519RceXG1ptNwb476Vz3OTe7nTmQ'

# Global variables for masking and image paths
rectangles = []
masked_image_path = ''
edited_image_path = ''
default_folder = 'fashion'  # Default folder to find similar images

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

def reset_mask():
    global rectangles
    rectangles = []
    canvas.delete("all")
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

def done_masking():
    create_masked_image(image_path)
    mask_window.destroy()

def create_masked_image(image_path):
    global masked_image_path

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Calculate the scale factor
    canvas_width, canvas_height = canvas.winfo_width(), canvas.winfo_height()
    img_width, img_height = img.shape[1], img.shape[0]
    scale_x = img_width / canvas_width
    scale_y = img_height / canvas_height

    for rect in rectangles:
        start_x, start_y, end_x, end_y = rect
        # Scale the coordinates back to the image size
        start_x = int(start_x * scale_x)
        start_y = int(start_y * scale_y)
        end_x = int(end_x * scale_x)
        end_y = int(end_y * scale_y)
        start_x, start_y = max(start_x, 0), max(start_y, 0)
        end_x, end_y = min(end_x, img.shape[1]), min(end_y, img.shape[0])
        img[start_y:end_y, start_x:end_x, 3] = 0  # Set selected region to fully transparent

    masked_image_path = 'output_image.png'
    cv2.imwrite(masked_image_path, img)
    print(f"Image saved to {masked_image_path}")

    return masked_image_path

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
    global edited_image_path

    host = f"https://api.stability.ai/v2beta/stable-image/edit/inpaint"

    params = {
        "image": image_path,
        "prompt": prompt,
        "strength": strength,
        "seed": seed,
        "output_format": output_format,
        "mode": "image-to-image",
        "model": "sd3-medium"
    }

    response = send_generation_request(host, params)

    output_image = response.content
    finish_reason = response.headers.get("finish-reason")
    seed = response.headers.get("seed")

    if finish_reason == 'CONTENT_FILTERED':
        raise Warning("Generation failed NSFW classifier")

    filename, _ = os.path.splitext(os.path.basename(image_path))
    edited_image_path = f"generated_{seed}.{output_format}"
    with open(edited_image_path, "wb") as f:
        f.write(output_image)
    print(f"Saved image {edited_image_path}")

    return edited_image_path

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def extract_features(model, img_path):
    img_array = load_and_preprocess_image(img_path)
    features = model.predict(img_array)
    return features.flatten()

def find_similar_images(reference_img_path, folder_path, top_n=9):
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

def load_image():
    global image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if image_path:
        img = Image.open(image_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        lbl_image.config(image=img_tk)
        lbl_image.image = img_tk
        show_masking_window()

def show_masking_window():
    global mask_window, canvas, img_tk, rectangles
    rectangles = []
    mask_window = tk.Toplevel(root)
    mask_window.title("Choose a section to mask")

    img = Image.open(image_path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    canvas = tk.Canvas(mask_window, width=img.width, height=img.height)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

    canvas.bind("<Button-1>", select_rectangle)
    canvas.bind("<B1-Motion>", update_rectangle)
    canvas.bind("<ButtonRelease-1>", save_rectangle)

    frame_buttons = tk.Frame(mask_window)
    frame_buttons.pack(fill=tk.X, padx=10, pady=10)

    btn_reset = ttk.Button(frame_buttons, text="Reset", command=reset_mask)
    btn_reset.pack(side=tk.LEFT, padx=5)

    btn_done = ttk.Button(frame_buttons, text="Done", command=done_masking)
    btn_done.pack(side=tk.RIGHT, padx=5)

def edit_image():
    prompt = text_prompt.get("1.0", tk.END).strip()
    if prompt:
        edit_image_with_stability(masked_image_path, prompt)
        img = Image.open(edited_image_path)
        img.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        lbl_edited_image.config(image=img_tk)
        lbl_edited_image.image = img_tk
        find_images()  # Automatically find similar images after editing

def find_images():
    similar_images = find_similar_images(edited_image_path, default_folder)
    for widget in frame_similar_images.winfo_children():
        widget.destroy()
    lbl_heading = tk.Label(frame_similar_images, text="Shop from Here", font=('Helvetica', 16, 'bold'))
    lbl_heading.grid(row=0, columnspan=3, pady=(0, 10))  # Add heading label
    for i, img_path in enumerate(similar_images):
        img = Image.open(img_path)
        img.thumbnail((100, 100))
        img_tk = ImageTk.PhotoImage(img)
        lbl = tk.Label(frame_similar_images, image=img_tk)
        lbl.image = img_tk
        lbl.grid(row=(i // 3) + 1, column=i % 3, padx=10, pady=10)  # Adjust row index

root = tk.Tk()
root.title("AI TRIAL ROOM")

# UI Elements
lbl_image = tk.Label(root)
lbl_image.grid(row=0, column=0, padx=10, pady=10)

lbl_edited_image = tk.Label(root)
lbl_edited_image.grid(row=0, column=1, padx=10, pady=10)

frame_similar_images = tk.Frame(root)
frame_similar_images.grid(row=0, column=2, padx=10, pady=10)

btn_load_image = ttk.Button(root, text="Load Image", command=load_image)
btn_load_image.grid(row=1, column=0, padx=10, pady=10)

btn_edit_image = ttk.Button(root, text="Edit Image", command=edit_image)
btn_edit_image.grid(row=3, column=1, padx=10, pady=10)

tk.Label(root, text="Prompt:").grid(row=2, column=0, padx=10, pady=10)
text_prompt = tk.Text(root, width=50, height=5)
text_prompt.grid(row=2, column=1, padx=10, pady=10)

root.mainloop()
