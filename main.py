from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load trained model
model = load_model("super_res_autoencoder.keras")

def preprocess_image(image_data):
    """Preprocess uploaded image for super-resolution."""
    image = Image.open(BytesIO(image_data)).convert("RGB")  # Convert to RGB
    image = image.resize((128, 128))  # Resize to match model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def postprocess_image(image_array):
    """Convert model output to a displayable image."""
    image_array = np.squeeze(image_array, axis=0)  # Remove batch dimension
    image_array = np.clip(image_array * 255.0, 0, 255).astype(np.uint8)  # Convert to uint8
    return Image.fromarray(image_array)

@app.get("/")
def read_root():
    return {"message": "Image Super-Resolution API is running!"}

@app.post("/super-resolve/")
async def super_resolve(image: UploadFile = File(...)):
    """Super-resolve an uploaded image."""
    image_data = await image.read()
    processed_image = preprocess_image(image_data)

    # Predict super-resolution image
    super_res_image = model.predict(processed_image)
    super_res_pil = postprocess_image(super_res_image)

    # Save and return image path (or convert to bytes and return as response)
    save_path = "output_super_res.png"
    super_res_pil.save(save_path)

    return {"message": "Super-resolution successful", "output_image": save_path}


