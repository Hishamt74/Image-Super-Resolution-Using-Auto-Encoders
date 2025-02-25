from tensorflow.keras.models import load_model
import numpy as np

# Load trained autoencoder model
model_path = "super_res_autoencoder.keras"  # Update with correct model path
autoencoder = load_model(model_path)

def super_resolve_image(image_array: np.ndarray) -> np.ndarray:
    """
    Predict high-resolution image from low-resolution input.
    """
    return autoencoder.predict(image_array)
