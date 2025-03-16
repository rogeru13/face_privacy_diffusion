import os
import cv2
import torch
import requests
import numpy as np
from retinaface import RetinaFace  # Face detection
from diffusers import StableDiffusionInpaintPipeline  # Flux1B (SD-based inpainting)
from PIL import Image
from io import BytesIO
import pandas as pd
import random
from huggingface_hub import login


# Set up directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# # Load the model with correct dtype and device settings
# pipe = StableDiffusionInpaintPipeline.from_pretrained (
#     "runwayml/stable-diffusion-inpainting", 
#     torch_dtype=torch.float16,  # Float16 is faster on GPU
#     force_download=True
# )

# # Move model to GPU
# pipe.to("cuda")

def detect_faces(image_path):
    """Detect faces using RetinaFace and return bounding boxes."""
    image = cv2.imread(image_path)
    faces = RetinaFace.detect_faces(image_path)
    if isinstance(faces, dict):
        return [(faces[k]['facial_area'], image) for k in faces.keys()]
    return []


def inpaint_faces(image_path, face_bboxes):
    """Replace detected faces with AI-generated ones using Flux1B."""
    image = Image.open(image_path).convert("RGB")
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    
    for (x1, y1, x2, y2), _ in face_bboxes:
        mask[y1:y2, x1:x2] = 255  # Create mask for inpainting
    
    mask = Image.fromarray(mask)
    inpainted_image = pipe(prompt="A realistic face", image=image, mask_image=mask).images[0]
    return inpainted_image


def download_image(url, save_path):
    """Download image from URL and save it."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image.save(save_path)
            return save_path
    except Exception as e:
        print(f"Failed to download {url}") #: {e}")
    return None

def process_image(url, img_id):
    """Download image, detect faces, replace them, and save result."""
    raw_path = f"data/raw/{img_id}.jpg"
    processed_path = f"data/processed/{img_id}.jpg"
    
    if download_image(url, raw_path):
        faces = detect_faces(raw_path)
        if faces:
            print(f"Face detected in {img_id}")
            # inpainted = inpaint_faces(raw_path, faces)
            # inpainted.save(processed_path)
        else:
            print(f"No faces detected in {img_id}")
            # image = Image.open(raw_path).convert("RGB")
            # image.save(processed_path)
    else:
        print(f"Skipping {img_id} due to download failure.")


# Replace with the path to your Parquet file
file_path = 'laion_sample.parquet'

# Read the Parquet file
df = pd.read_parquet("laion_sample.parquet")

image_urls = df["URL"].dropna().tolist()
sample_size = int(len(image_urls) * 0.001)  # Adjust based on needs
subset_urls = random.sample(image_urls, min(sample_size, 1000))  # Max 1000 images

for idx, url in enumerate(subset_urls[0:8]):
    process_image(url, f"img_{idx}")
