import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

st.title("🍽️ Food Image Classification (ResNet18)")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load class names
# ---------------------------
if not os.path.exists("class_names.txt"):
    st.error("❌ 'class_names.txt' not found! Add it to the same folder.")
    st.stop()

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

num_classes = len(class_names)

# ---------------------------
# Preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("best_model.pth"):
        st.error("❌ best_model.pth not found! Place it in this folder.")
        st.stop()

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model


model = load_model()

# ---------------------------
# Prediction Function
# ---------------------------
def predict(img):
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        confidence = torch.softmax(outputs, dim=1)[0][pred].item() * 100

    return class_names[pred], confidence


# ---------------------------
# Streamlit UI
# ---------------------------
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    label, conf = predict(img)

    st.success(f"🍛 Predicted: **{label}**")
    st.info(f"🔍 Confidence: **{conf:.2f}%**")
