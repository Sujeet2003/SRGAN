import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Now import other libraries
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import io
import RRDBNet_arch as arch


# Set device
device = torch.device('cpu')  # Change to 'cuda' if using GPU

# Load ESRGAN model
model_path = 'models/RRDB_ESRGAN_x4.pth'
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
model = model.to(device)

st.title("🖼️ ESRGAN Super-Resolution Image Enhancer")

# File uploader
uploaded_file = st.file_uploader("Upload a low-resolution image", type=["png", "jpg", "jpeg"])

# Initialize session state to store the enhanced image
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Enhance button
    if st.button("Enhance Image"):
        # Convert to OpenCV format
        img = np.array(image)
        img = img[:, :, [2, 1, 0]]  # RGB to BGR
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img).data.squeeze().float().cpu().clamp_(0, 1).numpy()

        # Convert to displayable format
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # BGR to RGB
        output_image = (output * 255.0).round().astype(np.uint8)

        # Save enhanced image in session state
        enhanced_pil_image = Image.fromarray(output_image)
        st.session_state.enhanced_image = enhanced_pil_image

        # Display result
        st.image(enhanced_pil_image, caption='Enhanced Image (x4)', use_container_width=True)

# Download button if image is enhanced
if st.session_state.enhanced_image is not None:
    buf = io.BytesIO()
    st.session_state.enhanced_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="Download Enhanced Image", data=byte_im, file_name="enhanced_image.png", mime="image/png")