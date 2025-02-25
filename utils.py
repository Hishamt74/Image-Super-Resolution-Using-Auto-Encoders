import numpy as np
from PIL import Image
from io import BytesIO
import base64
from tensorflow.keras.preprocessing.image import img_to_array

# Image dimensions
LOW_RES_HEIGHT = 128
LOW_RES_WIDTH = 128

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the input image to fit the model.
    """
    image = image.convert("RGB")  # Convert to RGB format
    image = image.resize((LOW_RES_WIDTH, LOW_RES_HEIGHT))  # Resize
    image_array = img_to_array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

def array_to_base64(image_array: np.ndarray) -> str:
    """
    Convert a NumPy image array to a base64-encoded PNG image.
    """
    image_array = (image_array[0] * 255).astype(np.uint8)  # Rescale to 0-255
    image_pil = Image.fromarray(image_array)
    
    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()
