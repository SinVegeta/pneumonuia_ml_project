# app.py
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('pneumonia_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🫁 Pneumonia Detection from Chest X-ray")
st.markdown("Upload a chest X-ray image, and the model will predict whether it shows **Normal** lungs or **Pneumonia**.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Chest X-ray', use_column_width=True)

    if st.button("Predict"):
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            prediction = "Normal" if predicted.item() == 0 else "Pneumonia"
        
        st.success(f"Prediction: **{prediction}**")
