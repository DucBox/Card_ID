import numpy as np

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def apply_nms(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to remove overlapping detections.
    Detections should be a list of tuples: [(bbox, confidence, label), ...]
    """
    print("[INFO] Applying NMS...")
    
    filtered_detections = {}
    unique_labels = set(label for _, _, label in detections)
    
    for label in unique_labels:
        label_detections = [det for det in detections if det[2] == label]
        label_detections.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence

        selected_boxes = []
        while label_detections:
            best_box = label_detections.pop(0)
            selected_boxes.append(best_box)
            
            label_detections = [
                box for box in label_detections
                if compute_iou(best_box[0], box[0]) < iou_threshold
            ]

        if selected_boxes:
            filtered_detections[label] = selected_boxes[0][0]
    
    print("[INFO] Filtered detections:", filtered_detections)
    return filtered_detections
