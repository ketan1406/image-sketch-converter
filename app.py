import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- Define the image processing functions ---


def colored_sketch(pil_img):
    # Convert PIL Image (RGB) to OpenCV BGR format
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    stylized = cv2.stylization(img_bgr, sigma_s=60, sigma_r=0.6)
    # Convert back to RGB for display in Streamlit
    return cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB)


def edge_sketch(pil_img):
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    grey_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grey_img, 50, 150)
    return edges


def cartoonify(pil_img):
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    grey_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.medianBlur(grey_img, 5)
    edges = cv2.adaptiveThreshold(blur_img, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color_img = cv2.bilateralFilter(
        img_bgr, d=9, sigmaColor=300, sigmaSpace=300)
    cartoon = cv2.bitwise_and(color_img, color_img, mask=edges)
    return cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)


def adjustable_sketch(pil_img, blur_ksize=111):
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    grey_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    invert_img = cv2.bitwise_not(grey_img)
    # Ensure the kernel size is odd and greater than 1
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blur_img = cv2.GaussianBlur(invert_img, (blur_ksize, blur_ksize), 0)
    invblur_img = cv2.bitwise_not(blur_img)
    sketch_img = cv2.divide(grey_img, invblur_img, scale=256.0)
    return sketch_img

# --- Helper function to convert a NumPy array to a downloadable image ---


def convert_array_to_bytes(result):
    if len(result.shape) == 2:  # Grayscale image
        pil_img = Image.fromarray(result)
    else:
        pil_img = Image.fromarray(result)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# --- Streamlit App UI ---
st.title("Image Converter App")
st.write("Upload an image and choose a sketch effect to transform it.")

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image with PIL and display it
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Let user select a sketch style
    sketch_style = st.selectbox("Select Sketch Style",
                                ["Pencil Sketch (B/W)", "Colored Sketch", "Edge-Detection Sketch", "Cartoon Effect"])

    # Process the image based on user selection
    if sketch_style == "Pencil Sketch (B/W)":
        # Optional: let the user adjust the blur kernel size with a slider
        blur_ksize = st.slider("Adjust Blur Kernel Size (odd numbers)",
                               min_value=3, max_value=201, step=2, value=111)
        result = adjustable_sketch(image, blur_ksize=blur_ksize)
        st.image(result, caption="Pencil Sketch",
                 use_column_width=True, clamp=True, channels="GRAY")
    elif sketch_style == "Colored Sketch":
        result = colored_sketch(image)
        st.image(result, caption="Colored Sketch", use_column_width=True)
    elif sketch_style == "Edge-Detection Sketch":
        result = edge_sketch(image)
        st.image(result, caption="Edge-Detection Sketch",
                 use_column_width=True, clamp=True, channels="GRAY")
    elif sketch_style == "Cartoon Effect":
        result = cartoonify(image)
        st.image(result, caption="Cartoon Effect", use_column_width=True)
    else:
        st.error("Invalid choice, defaulting to Pencil Sketch.")
        result = adjustable_sketch(image)
        st.image(result, caption="Pencil Sketch",
                 use_column_width=True, clamp=True, channels="GRAY")

    # Provide a download button for the processed image
    st.markdown("---")
    st.write("Download your sketch:")
    img_bytes = convert_array_to_bytes(result)
    st.download_button(label="Download Image",
                       data=img_bytes,
                       file_name="sketch.png",
                       mime="image/png")
