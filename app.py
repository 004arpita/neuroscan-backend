import os
import requests
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Step 1: Define the model path and URL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.keras')
MODEL_URL = "https://drive.google.com/uc?export=download&id=1TOUeYxLknhScvVEjbZknfiGn1kabspvv"

# Step 2: Download the model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    try:
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Failed to download model: {e}")

# Step 3: Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Keras model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None
