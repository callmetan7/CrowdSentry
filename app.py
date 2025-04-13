import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
from src.model import MCNN
from src.interpretation import generateRisk
import matplotlib.pyplot as plt
import pandas as pd
import geocoder
import os

location = (geocoder.ip('me')).city

# Function to load the model
def load_model(model_path, device):
    model = MCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Function to preprocess the input image
def preprocess_image(image, target_size=(224, 224)):
    if isinstance(image, Image.Image):
        image = np.array(image)  # Convert PIL.Image to numpy array
    image_resized = cv2.resize(image, target_size)
    image_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    return image_tensor

# Function to predict the headcount
def predict_headcount(image_tensor, model, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        density_map = model(image_tensor).cpu().squeeze(0).numpy()
    crowd_count = np.sum(density_map)
    return crowd_count, density_map

# Function to visualize the density map
def visualize_density_map(density_map, predicted_count):
    """
    Visualizes the density map using matplotlib.

    Args:
        density_map (numpy.ndarray): Predicted density map.
        predicted_count (float): Total predicted crowd count.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(density_map.squeeze(), cmap="jet")  # Remove channel dimension
    plt.colorbar()
    plt.title(f"Predicted Density Map (Count: {predicted_count:.2f})")
    plt.axis("off")
    st.pyplot(plt)

# Streamlit UI
st.title("Crowd Counting with MCNN Models")
st.write("Upload an image and select a pre-trained model to predict the crowd count.")
 
event_type = st.selectbox("Event Type", ["Concert", "Pilgrimage", "Protest"])
percentExitsBlocked = st.slider("Percentage of Exits Blocked", 0, 100, 0, 1, label_visibility="visible")
# Sidebar for model selection
model_dir = "models"  # Directory where models are stored
if not os.path.exists(model_dir):
    st.error(f"Model directory '{model_dir}' not found!")
    st.stop()

model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
if not model_files:
    st.error("No models found in the directory!")
    st.stop()

selected_model = st.sidebar.selectbox("Select a Model", model_files)

# Image upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform prediction on button click
    if st.button("Predict Headcount"):
        st.write("Processing...")

        # Preprocess the image
        image_tensor = preprocess_image(image)

        # Load the selected model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(model_dir, selected_model)
        model = load_model(model_path, device)

        # Predict headcount
        predicted_count, density_map = predict_headcount(image_tensor, model, device)

        # Display results
        st.success(f"Predicted Crowd Count: {int(predicted_count)}")
        st.write("Visualizing density map...")
        visualize_density_map(density_map, predicted_count)

        # Display Interpretation
        st.write("Interpretation:")
        print(np.average(density_map))
        st.write(generateRisk(
            density = np.average(density_map),
            location= geocoder.ip('me').city,
            eventType = event_type.lower(),
            percentExitsBlocked = int(percentExitsBlocked)
        ))