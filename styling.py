from io import BytesIO
import IPython
import json
import os
from PIL import Image
import requests
import time
import getpass

STABILITY_KEY = 'sk-EZvd1qLoQfO8kqH00ILWcSbcvKCD4GGbpgJuCcdR5s47o7M1'

def send_generation_request(
    host,
    params,
):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    # Encode parameters
    files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files)==0:
        files["none"] = ''

    # Send request
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

image = "output_image.png" #@param {type:"string"}
prompt = f"Change the model's outfit to a beautiful traditional saree" #@param {type:"string"}
seed = 0 #@param {type:"integer"}
output_format = "jpeg" #@param ["jpeg", "png"]
strength = 0.75 #@param {type:"slider", min:0.0, max: 1.0, step: 0.01}

host = f"https://api.stability.ai/v2beta/stable-image/edit/inpaint"

params = {
        "image" : image,
        "prompt" : prompt,
        "strength" : strength,
        "seed" : seed,
        "output_format": output_format,
        "mode" : "image-to-image",
        "model" : "sd3-medium"
}

response = send_generation_request(
    host,
    params
)

# Decode response
output_image = response.content
finish_reason = response.headers.get("finish-reason")
seed = response.headers.get("seed")

# Check for NSFW classification
if finish_reason == 'CONTENT_FILTERED':
    raise Warning("Generation failed NSFW classifier")

# Save and display result
generated = f"generated_{seed}.{output_format}"
with open(generated, "wb") as f:
    f.write(output_image)
print(f"Saved image {generated}")

# output.no_vertical_scroll()
print("Result image:")
IPython.display.display(Image.open(generated))