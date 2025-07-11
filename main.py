import streamlit as st
import torch
import numpy as np
from PIL import Image
from RRDBNet_arch import RRDBNet
import os

# --- Model configuration ---
model_path = 'models/RRDB_PSNR_x4.pth'  # path to the ESRGAN pretrained model
device = torch.device('cpu')

# Load the model
def load_model():
    model = RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model

# Preprocess input image
def preprocess_image(img: Image.Image):
    img = np.array(img).astype(np.float32)
    img = img / 255.
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)
    return img_LR

# Postprocess output image
def postprocess_image(img_SR):
    img_SR = img_SR.squeeze().float().cpu().clamp_(0, 1).numpy()
    img_SR = np.transpose(img_SR, (1, 2, 0))
    img_SR = (img_SR * 255.0).round().astype(np.uint8)
    return img_SR

# --- Streamlit UI ---
st.title("Image Super-Resolution using SRGAN")

uploaded_file = st.file_uploader("Upload a low-resolution image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Low-Resolution Input", use_container_width=True)

    if st.button("Enhance Image"):
        with st.spinner("Enhancing..."):
            model = load_model()
            img_LR = preprocess_image(image)
            with torch.no_grad():
                output = model(img_LR)
            result_img = postprocess_image(output)
            st.image(result_img, caption="Enhanced Image", use_container_width=True)
            st.success("Enhancement complete!")
