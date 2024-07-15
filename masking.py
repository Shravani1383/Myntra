import cv2
import numpy as np
import os

# Function to draw rectangles on the image
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

# Path to the image
image_path = 'family.jpg'  # Replace with your image path

# Check if the image path exists
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image file '{image_path}' not found.")

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

refPt = []
rectangles = []
cropping = False

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
