import cv2
from ultralytics import YOLO
from src.utils import apply_nms

def filter_text_boxes(text_boxes):
    """
    Lọc bỏ các vùng không cần thiết (các title như 'id_title', 'name_title', 'birth_title').
    Chỉ giữ lại 'id', 'name', 'birth'.
    """
    valid_labels = {"id", "name", "birth"}
    filtered_boxes = {label: coords for label, coords in text_boxes.items() if label in valid_labels}

    print("[INFO] Filtered text fields:", filtered_boxes)
    return filtered_boxes

def detect_text_regions(model, image, iou_threshold=0.5):
    """
    Nhận diện các vùng chứa văn bản trên ảnh bằng YOLO.
    """
    print("[INFO] Running YOLO model for text detection...")
    result = model(image)

    raw_detections = [
        ((int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])),
         box.conf[0].item(), model.names[int(box.cls[0].item())])
        for box in result[0].boxes
    ]

    print("[INFO] Raw detected text regions:", raw_detections)
    filtered_text_boxes = apply_nms(raw_detections, iou_threshold)

    # Lọc bỏ các class title
    return filter_text_boxes(filtered_text_boxes)

