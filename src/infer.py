import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import matplotlib.pyplot as plt
from src.card_detection import load_yolo_model, detect_corners
from src.transform_card import perspective_transform
from src.text_detection import detect_text_regions
from src.text_recognition import load_vietocr, extract_text_from_boxes

# Đường dẫn mô hình YOLO 
MODEL_PATH = "../models/card_detect.pt"
TEXT_MODEL_PATH = "../models/text_recog.pt"

# Đường dẫn ảnh test (Cập nhật ảnh cụ thể)
IMAGE_PATH = "/Users/ngoquangduc/Desktop/AI_Project/Card_ID/data/raw/IMG_2215.JPG"

def main():
    print("[INFO] Loading YOLO model...")
    model = load_yolo_model(MODEL_PATH)

    print(f"[INFO] Loading image: {IMAGE_PATH}")
    image = cv2.imread(IMAGE_PATH)
    
    if image is None:
        print("[ERROR] Failed to load image. Check the file path.")
        return

    print("[INFO] Running corner detection...")
    filtered_corners, corner_centers = detect_corners(model, image)

    print("[INFO] Detected corners:", filtered_corners)
    
    if filtered_corners:
        transformed_image = perspective_transform(image, corner_centers)
    
    text_model = load_yolo_model(TEXT_MODEL_PATH)
    # Nhận diện vùng văn bản
    if transformed_image is not None:
        text_boxes = detect_text_regions(text_model, transformed_image)

    vietocr_detector = load_vietocr()
    
    if transformed_image is not None and text_boxes:
        extracted_texts = extract_text_from_boxes(transformed_image, text_boxes, vietocr_detector)

        # In kết quả cuối cùng
        print("\n[RESULT] Extracted Information:")
        print(f"ID: {extracted_texts.get('id', 'N/A')}")
        print(f"Name: {extracted_texts.get('name', 'N/A')}")
        print(f"Birth: {extracted_texts.get('birth', 'N/A')}")      

if __name__ == "__main__":
    main()
