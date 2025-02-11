import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.config import CORNER_MODEL_PATH, TEXT_MODEL_PATH
from src.card_detection import load_yolo_model, detect_corners
from src.transform_card import perspective_transform
from src.text_detection import detect_text_regions
from src.text_recognition import load_vietocr, extract_text_from_boxes

# Load models từ config

st.write(f"Loading model from {CORNER_MODEL_PATH} and {TEXT_MODEL_PATH})

corner_model = load_yolo_model(CORNER_MODEL_PATH)
text_model = load_yolo_model(TEXT_MODEL_PATH)
ocr_model = load_vietocr()

st.set_page_config(page_title="ID Card OCR", layout="centered")

st.title("🆔 ID Card Recognition")

uploaded_file = st.file_uploader("📤 Upload ID Card Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Đọc ảnh
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Nhận diện góc
    st.write("🔍 Detecting card corners...")
    filtered_corners, corner_centers = detect_corners(corner_model, image)
    
    # Transform perspective
    if filtered_corners:
        st.write("📐 Transforming perspective...")
        transformed_image = perspective_transform(image, corner_centers)
    else:
        st.error("🚨 Failed to detect corners!")
        st.stop()

    # Nhận diện vùng chứa text
    st.write("📝 Detecting text regions...")
    text_boxes = detect_text_regions(text_model, transformed_image)

    # Nhận diện văn bản
    if text_boxes:
        st.write("🔠 Extracting text...")
        extracted_texts = extract_text_from_boxes(transformed_image, text_boxes, ocr_model)

        # Hiển thị kết quả
        st.subheader("📋 Extracted Information")
        st.write(f"**ID:** {extracted_texts.get('id', 'N/A')}")
        st.write(f"**Name:** {extracted_texts.get('name', 'N/A')}")
        st.write(f"**Birth:** {extracted_texts.get('birth', 'N/A')}")
    else:
        st.error("🚨 No text detected!")
