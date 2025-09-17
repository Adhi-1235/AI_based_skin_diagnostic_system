import pathlib
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import requests
import zipfile

st.set_page_config(page_title="Skin Disease Classifier", layout="centered")
st.title("🩺 AI Skin Disease Classifier — MobileNetV2")
st.caption("Educational demo — not a medical device. Always consult a dermatologist.")

# Paths
OUTPUT_DIR = pathlib.Path("outputs")
MODEL_PATH = OUTPUT_DIR / "mobilenetv2_ham10000.keras"
CLASS_INDEX_PATH = OUTPUT_DIR / "class_indices.json"

# Function to download and extract outputs.zip if outputs folder does not exist
def download_file_from_google_drive(id, destination):
    """Download a large file from Google Drive by handling the confirmation token."""
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

def download_and_extract_outputs():
    outputs_folder = 'outputs'
    zip_path = 'outputs.zip'
    gdrive_file_id = '1CM6-mKThotW89oXLed0KmXf-f5MlUkPO'
    if not os.path.exists(outputs_folder):
        download_file_from_google_drive(gdrive_file_id, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(zip_path)

# Load model & classes
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(CLASS_INDEX_PATH, "r") as f:
        idx2class = json.load(f)
    idx2class = {int(k): v for k, v in idx2class.items()}
    return model, idx2class

download_and_extract_outputs()
model, idx2class = load_artifacts()
class_names = [idx2class[i] for i in sorted(idx2class.keys())]

# Image preprocessing
def load_and_prep_image(img_bytes, target_size=(224,224)):
    img = Image.open(img_bytes).convert("RGB").resize(target_size)
    x = np.array(img).astype("float32")
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x, img

# Educational medicine recommendations
def safe_recommendation(label):
    suggestions = {
        "nv": "Likely benign melanocytic nevus. Monitor changes (ABCDE). Use sunscreen SPF 30+. See dermatologist if evolving.",
        "mel": "Possible melanoma — urgent dermatologist evaluation recommended. Do not self-medicate.",
        "bkl": "Benign keratosis-like lesion. Usually harmless; consider dermatologist consult if symptomatic.",
        "bcc": "Basal cell carcinoma — seek prompt dermatologist evaluation.",
        "akiec": "Actinic keratosis / intraepithelial carcinoma — dermatologist evaluation recommended.",
        "vasc": "Vascular lesion (e.g., angioma). Usually benign; consult if bleeding or rapid change.",
        "df": "Dermatofibroma. Often benign; consult dermatologist if symptomatic or changing."
    }
    return suggestions.get(label, "Consult a qualified dermatologist for personalized advice.")

# UI
uploaded = st.file_uploader("Upload a skin lesion image (JPG/PNG)", type=["jpg","jpeg","png"])
if uploaded:
    x, img = load_and_prep_image(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)
    
    with st.spinner("Analyzing..."):
        probs = model.predict(x)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]
        confidence = float(probs[pred_idx])

    st.subheader(f"Prediction: **{pred_label}**")
    st.write(f"Confidence: {confidence:.2%}")

    # Probability table
    prob_table = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    st.write(pd.DataFrame([prob_table]).T.rename(columns={0:"probability"}))

    # Educational guidance
    st.divider()
    st.subheader("Recommended Action (Educational)")
    st.write(safe_recommendation(pred_label))
    st.info("This app is for educational purposes only and is **not** a medical diagnosis or prescription.")
else:
    st.write("Please upload a lesion image to get a prediction.")


