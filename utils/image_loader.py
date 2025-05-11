from PIL import Image
import os

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Could not load image: {e}")

