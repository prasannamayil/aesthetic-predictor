import os
import csv
import random
import requests
import torch
import torch.nn as nn
from PIL import Image
import open_clip
from os.path import expanduser
from urllib.request import urlretrieve
import io

def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = os.path.join(home, ".cache", "emb_reader")
    path_to_model = os.path.join(cache_folder, f"sa_0_4_{clip_model}_linear.pth")
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/"
            f"sa_0_4_{clip_model}_linear.pth?raw=true"
        )
        print(f"Downloading aesthetic model to {path_to_model}...")
        try:
            urlretrieve(url_model, path_to_model)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None
    try:
        if clip_model == "vit_l_14":
            m = nn.Linear(768, 1)
        elif clip_model == "vit_b_32":
            m = nn.Linear(512, 1)
        else:
            raise ValueError("Unsupported CLIP model type")

        # Load weights with appropriate map_location
        if torch.cuda.is_available():
            s = torch.load(path_to_model)
        else:
            s = torch.load(path_to_model, map_location=torch.device('cpu'))

        m.load_state_dict(s)
        m.eval()
        return m
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None

def get_image_urls(csv_filepath, sample_size=50):
    """Reads image URLs from the CSV and returns a random sample."""
    urls = []
    try:
        with open(csv_filepath, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 2 and row[1].startswith('http'):
                    urls.append(row[1])
        if len(urls) < sample_size:
             print(f"Warning: Only found {len(urls)} URLs, sampling all of them.")
             return urls
        return random.sample(urls, sample_size)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
        return []
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

def download_image(url):
    """Downloads an image from a URL and returns a PIL Image object."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes
        image_bytes = response.content
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return pil_image
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return None
    except Exception as e:
        print(f"Error processing image from {url}: {e}")
        return None

def get_aesthetic_score(pil_image, clip_model, aesthetic_model, preprocess_fn, device):
    """Calculates the aesthetic score for a given PIL image."""
    if pil_image is None:
        return None
    try:
        # Preprocess the image and move it to the correct device
        image_processed = preprocess_fn(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            # Encode image features using CLIP
            image_features = clip_model.encode_image(image_processed)
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # Predict aesthetic score using the linear model
            prediction = aesthetic_model(image_features)
        return prediction.item() # Return score as a float
    except Exception as e:
        print(f"Error calculating score: {e}")
        return None