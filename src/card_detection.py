import cv2
from ultralytics import YOLO
from src.utils import apply_nms

def detect_corners(model, image, iou_threshold=0.5):
    """
    Detect the four corners of the ID card using YOLO.
    """
    print("[INFO] Running YOLO model for corner detection...")
    result = model(image)
    
    raw_detections = [
        ((int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])),
         box.conf[0].item(), model.names[int(box.cls[0].item())])
        for box in result[0].boxes
    ]
    
    print("[INFO] Raw detected corners:", raw_detections)
    filtered_corners = apply_nms(raw_detections, iou_threshold)

    corner_centers = {
        label: ((coords[0] + coords[2]) // 2, (coords[1] + coords[3]) // 2)
        for label, coords in filtered_corners.items()
    }

    print("[INFO] Final corner centers:", corner_centers)
    return filtered_corners, corner_centers

def load_yolo_model(model_path):
    print(f"[INFO] Loading YOLO model from {model_path}...")
    return YOLO(model_path)
