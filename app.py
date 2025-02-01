import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# ---------------------------------------------------------------------------------
# 🎨 Image Processing Functions
# ---------------------------------------------------------------------------------


def colored_sketch(pil_img, sigma_s=60, sigma_r=0.6):
    # Convert the input PIL image to OpenCV BGR format
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    #  Apply stylization with adjustable sigma_s and sigma_r parameters
    stylized = cv2.stylization(img_bgr, sigma_s=sigma_s, sigma_r=sigma_r)
    #  Convert the stylized image back to RGB for display
    return cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB)


def edge_sketch(pil_img, lower_threshold=50, upper_threshold=150):
    #  Convert image to BGR then grayscale
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    grey_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    #  Apply Canny edge detection with adjustable thresholds
    edges = cv2.Canny(grey_img, lower_threshold, upper_threshold)
    return edges


def cartoonify(pil_img, median_blur_ksize=5, bilateral_d=9, bilateral_sigmaColor=300, bilateral_sigmaSpace=300,
               adaptive_thresh_blockSize=9, adaptive_thresh_C=9):
    #  Convert image to BGR format
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    grey_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    #  Ensure median blur kernel size is odd
    if median_blur_ksize % 2 == 0:
        median_blur_ksize += 1
    blur_img = cv2.medianBlur(grey_img, median_blur_ksize)

    #  Ensure adaptive threshold block size is odd (required by adaptiveThreshold)
    if adaptive_thresh_blockSize % 2 == 0:
        adaptive_thresh_blockSize += 1
    edges = cv2.adaptiveThreshold(blur_img, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, adaptive_thresh_blockSize, adaptive_thresh_C)

    #  Apply bilateral filter to smooth colors and combine with edges
    color_img = cv2.bilateralFilter(img_bgr, d=bilateral_d,
                                    sigmaColor=bilateral_sigmaColor, sigmaSpace=bilateral_sigmaSpace)
    cartoon = cv2.bitwise_and(color_img, color_img, mask=edges)
    return cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)


def adjustable_sketch(pil_img, blur_ksize=111, scale=256.0):
    #  Convert image to BGR and then grayscale for pencil sketch effect
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    grey_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    invert_img = cv2.bitwise_not(grey_img)

    #  Ensure Gaussian blur kernel size is odd
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blur_img = cv2.GaussianBlur(invert_img, (blur_ksize, blur_ksize), 0)
    invblur_img = cv2.bitwise_not(blur_img)
    #  Create pencil sketch effect using cv2.divide with adjustable scale
    sketch_img = cv2.divide(grey_img, invblur_img, scale=scale)
    return sketch_img

# ---------------------------------------------------------------------------------
# 📥 Helper Function: Convert NumPy Array to Downloadable Image Bytes
# ---------------------------------------------------------------------------------


def convert_array_to_bytes(result):
    #  Convert NumPy array to a PIL Image, then save to a byte stream
    pil_img = Image.fromarray(result)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# ---------------------------------------------------------------------------------
# 🚀 Streamlit App UI
# ---------------------------------------------------------------------------------


st.title("🎨 Image Converter App")
st.write("Upload an image and choose a sketch effect to transform it! 😎")

#  Image uploader widget
uploaded_file = st.file_uploader(
    "📁 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    #  Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    st.markdown("## Choose Your Sketch Style")
    sketch_style = st.selectbox(
        "🖌️ Select Sketch Style",
        ["Pencil Sketch (B/W)", "Colored Sketch",
         "Edge-Detection Sketch", "Cartoon Effect"]
    )

    if sketch_style == "Pencil Sketch (B/W)":
        st.markdown("#### Pencil Sketch Settings")
        blur_ksize = st.slider("Adjust Blur Kernel Size (odd numbers)",
                               min_value=3, max_value=201, step=2, value=111)
        scale = st.slider("Adjust Sketch Scale", min_value=100.0,
                          max_value=300.0, step=1.0, value=256.0)
        result = adjustable_sketch(image, blur_ksize=blur_ksize, scale=scale)
        st.image(result, caption="✏️ Pencil Sketch",
                 use_container_width=True, clamp=True, channels="GRAY")

    elif sketch_style == "Colored Sketch":
        st.markdown("#### Colored Sketch Settings")
        sigma_s = st.slider("Sigma S (Stylization)",
                            min_value=10, max_value=100, step=1, value=60)
        sigma_r = st.slider("Sigma R (Stylization)",
                            min_value=0.1, max_value=1.0, step=0.1, value=0.6)
        result = colored_sketch(image, sigma_s=sigma_s, sigma_r=sigma_r)
        st.image(result, caption="🌈 Colored Sketch", use_container_width=True)

    elif sketch_style == "Edge-Detection Sketch":
        st.markdown("#### Edge-Detection Settings")
        lower_threshold = st.slider(
            "Lower Threshold", min_value=0, max_value=255, value=50)
        upper_threshold = st.slider(
            "Upper Threshold", min_value=0, max_value=255, value=150)
        result = edge_sketch(
            image, lower_threshold=lower_threshold, upper_threshold=upper_threshold)
        st.image(result, caption="🔍 Edge-Detection Sketch",
                 use_container_width=True, clamp=True, channels="GRAY")

    elif sketch_style == "Cartoon Effect":
        st.markdown("#### Cartoon Effect Settings")
        median_blur_ksize = st.slider(
            "Median Blur Kernel Size", min_value=3, max_value=15, step=2, value=5)
        bilateral_d = st.slider("Bilateral Filter d",
                                min_value=1, max_value=20, step=1, value=9)
        bilateral_sigmaColor = st.slider(
            "Bilateral Sigma Color", min_value=50, max_value=500, step=10, value=300)
        bilateral_sigmaSpace = st.slider(
            "Bilateral Sigma Space", min_value=50, max_value=500, step=10, value=300)
        adaptive_thresh_blockSize = st.slider(
            "Adaptive Threshold Block Size", min_value=3, max_value=21, step=2, value=9)
        adaptive_thresh_C = st.slider(
            "Adaptive Threshold C", min_value=0, max_value=20, step=1, value=9)
        result = cartoonify(image, median_blur_ksize=median_blur_ksize, bilateral_d=bilateral_d,
                            bilateral_sigmaColor=bilateral_sigmaColor, bilateral_sigmaSpace=bilateral_sigmaSpace,
                            adaptive_thresh_blockSize=adaptive_thresh_blockSize, adaptive_thresh_C=adaptive_thresh_C)
        st.image(result, caption="😎 Cartoon Effect", use_container_width=True)

    else:
        st.error("❌ Invalid choice, defaulting to Pencil Sketch.")
        result = adjustable_sketch(image)
        st.image(result, caption="✏️ Pencil Sketch",
                 use_container_width=True, clamp=True, channels="GRAY")

    st.markdown("---")
    st.write("⬇️ **Download Your Sketch:**")
    img_bytes = convert_array_to_bytes(result)
    st.download_button(label="💾 Download Image",
                       data=img_bytes,
                       file_name="sketch.png",
                       mime="image/png")
