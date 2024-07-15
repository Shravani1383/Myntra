import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import requests

STABILITY_KEY = 'sk-EZvd1qLoQfO8kqH00ILWcSbcvKCD4GGbpgJuCcdR5s47o7M1'

# Global variables for masking
refPt = []
rectangles = []
cropping = False
image = None
image_copy = None

def select_rectangle(event):
    global refPt, cropping, image, image_copy, rectangles, canvas, tk_image

    x, y = event.x, event.y

    if event.type == tk.EventType.ButtonPress:
        refPt = [(x, y)]
        cropping = True

    elif event.type == tk.EventType.Motion and cropping:
        image_copy = image.copy()
        for rect in rectangles:
            cv2.rectangle(image_copy, rect[0], rect[1], (0, 255, 0), 2)
        cv2.rectangle(image_copy, refPt[0], (x, y), (0, 255, 0), 2)
        update_canvas()

    elif event.type == tk.EventType.ButtonRelease:
        refPt.append((x, y))
        cropping = False
        rectangles.append((refPt[0], refPt[1]))
        for rect in rectangles:
            cv2.rectangle(image_copy, rect[0], rect[1], (0, 255, 0), 2)
        update_canvas()

def update_canvas():
    global image_copy, tk_image, canvas

    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGRA2RGBA)
    pil_image = Image.fromarray(image_rgb)
    tk_image = ImageTk.PhotoImage(pil_image)
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    canvas.image = tk_image  # To prevent garbage collection

def create_masked_image(image_path):
    global image, image_copy, rectangles, canvas, tk_image

    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if the image is loaded properly
    if image is None:
        raise ValueError(f"Failed to load image '{image_path}'.")

    # If the image does not have an alpha channel, add one
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    image_copy = image.copy()

    # Set up the canvas
    image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGRA2RGBA)
    pil_image = Image.fromarray(image_rgb)
    tk_image = ImageTk.PhotoImage(pil_image)

    canvas.config(width=tk_image.width(), height=tk_image.height())
    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
    canvas.bind("<ButtonPress-1>", select_rectangle)
    canvas.bind("<B1-Motion>", select_rectangle)
    canvas.bind("<ButtonRelease-1>", select_rectangle)

    canvas.image = tk_image  # To prevent garbage collection

def finish_masking():
    global image, rectangles

    if not rectangles:
        messagebox.showerror("Error", "No regions selected for masking")
        return

    # Make the selected regions transparent
    for rect in rectangles:
        startX, startY = rect[0]
        endX, endY = rect[1]

        # Ensure coordinates are within the image boundaries
        startX, startY = max(startX, 0), max(startY, 0)
        endX, endY = min(endX, image.shape[1]), min(endY, image.shape[0])

        # Make the selected region transparent
        image[startY:endY, startX:endX, 3] = 0  # Set alpha channel to 0

    # Save the modified image
    output_path = 'output_image.png'
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

    app.masked_image_path = output_path

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

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Editor")

        self.image_path = None
        self.prompt = None
        self.masked_image_path = None

        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack()

        self.mask_button = tk.Button(root, text="Create Mask", command=self.create_mask)
        self.mask_button.pack()

        self.prompt_label = tk.Label(root, text="Edit Prompt:")
        self.prompt_label.pack()
        self.prompt_entry = tk.Entry(root)
        self.prompt_entry.pack()

        self.finish_button = tk.Button(root, text="Finish Masking", command=self.finish_masking)
        self.finish_button.pack()

        self.edit_button = tk.Button(root, text="Edit Image", command=self.edit_image)
        self.edit_button.pack()

        global canvas
        canvas = tk.Canvas(root)
        canvas.pack()

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            img = Image.open(self.image_path)
            img.thumbnail((300, 300))
            img = ImageTk.PhotoImage(img)
            canvas.config(width=img.width(), height=img.height())
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.image = img

    def create_mask(self):
        if not self.image_path:
            messagebox.showerror("Error", "No image loaded")
            return
        create_masked_image(self.image_path)

    def finish_masking(self):
        finish_masking()

    def edit_image(self):
        if not self.masked_image_path:
            messagebox.showerror("Error", "No masked image created")
            return
        self.prompt = self.prompt_entry.get()
        if not self.prompt:
            messagebox.showerror("Error", "No prompt provided")
            return
        try:
            edited_image_path = edit_image_with_stability(self.masked_image_path, self.prompt)
            img = Image.open(edited_image_path)
            img.thumbnail((300, 300))
            img = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=tk.NW, image=img)
            canvas.image = img
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()
