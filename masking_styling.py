import cv2
import numpy as np
import os
from io import BytesIO
import IPython
import requests
from PIL import Image

STABILITY_KEY = 'sk-EZvd1qLoQfO8kqH00ILWcSbcvKCD4GGbpgJuCcdR5s47o7M1'

# Global variables for masking
refPt = []
rectangles = []
cropping = False

def select_rectangle(event, x, y, flags, param):
    global refPt, cropping, image, image_copy, rectangles
    
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            image_copy = image.copy()
            for rect in rectangles:
                cv2.rectangle(image_copy, rect[0], rect[1], (0, 255, 0), 2)
            cv2.rectangle(image_copy, refPt[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("image", image_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        rectangles.append((refPt[0], refPt[1]))
        for rect in rectangles:
            cv2.rectangle(image_copy, rect[0], rect[1], (0, 255, 0), 2)
        cv2.imshow("image", image_copy)

def create_masked_image(image_path):
    global image, image_copy, rectangles
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if the image is loaded properly
    if image is None:
        raise ValueError(f"Failed to load image '{image_path}'.")

    # If the image does not have an alpha channel, add one
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

    image_copy = image.copy()

    # Set mouse callback function
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", select_rectangle)

    # Display image and wait for user input
    while True:
        cv2.imshow("image", image_copy)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("r"):
            image_copy = image.copy()
            rectangles = []
        
        elif key == ord("q"):
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()

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

# Main flow
if __name__ == "__main__":
    image_path = 'family.jpg'  # Replace with your image path
    prompt = "Change the outfit of the family in yellow outfits"

    # Step 1: Create masked image
    masked_image_path = create_masked_image(image_path)

    # Step 2: Edit image using Stability AI
    edited_image_path = edit_image_with_stability(masked_image_path, prompt)

    # Display the original and edited images
    print("Original image:")
    IPython.display.display(Image.open(image_path))
    print("Masked image:")
    IPython.display.display(Image.open(masked_image_path))
    print("Edited image:")
    IPython.display.display(Image.open(edited_image_path))
