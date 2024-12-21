
---

# Myntra AI Trial Room

The **Myntra AI Trial Room** leverages the Stability API to provide an innovative virtual fitting experience. This technology enables customers to try on clothing virtually, simulating how garments fit and look before purchase. By combining computer vision and augmented reality, the trial room accurately overlays clothing items onto live video feeds or uploaded images, offering a seamless and interactive shopping journey.

## Features
- **Virtual Fitting**: Try clothing virtually on uploaded images.
- **Mask Editing**: Select areas of the image for masking and editing.
- **Image Similarity**: Suggest similar items based on edited images.
- **Integration**: Powered by Stability AI for inpainting and VGG16 for feature extraction.

---

## Setup

### Prerequisites
Install the required Python libraries:

```bash
pip install keras scikit-learn tkinter keras-models keras-utils Keras-Applications
```

### API Key Setup
Add your Stability API key:
Visit the Stability AI platform: https://platform.stability.ai/.

```python
STABILITY_KEY = 'YOUR_API_KEY'
```

---

## Usage

Run the application:

```bash
python app.py
```

### Application Workflow
1. **Load an Image**: Upload an image to try on outfits.
2. **Mask Editing**: Select regions to modify and apply a custom prompt.
3. **Inpainting**: Use Stability AI to generate edited images based on your prompt.
4. **Find Similar Items**: Automatically suggest clothing items similar to the edited image.

---

## Core Code Overview

### Masking and Inpainting
- Allows users to select regions on the image to mask.
- Leverages Stability API to perform inpainting based on the prompt.

### Similarity Search
- Uses VGG16 pre-trained model to extract features from the edited image.
- Finds similar clothing items from a predefined dataset.

### UI Features
- **Tkinter-based GUI**: Interactive interface for seamless user interaction.
- **Real-time Updates**: Displays edited images and suggestions dynamically.

---

## Example Prompts
- "A summer dress with floral patterns."
- "A formal suit for a business meeting."
- "Casual wear with bright colors."

---

## Folder Structure
- **`fashion/`**: Default folder containing sample clothing images for similarity search.
- **Generated Images**: Saved in the application directory with the prefix `generated_`.

---

## Contribution
Feel free to fork and contribute to this project. For any issues, open a pull request or issue.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

