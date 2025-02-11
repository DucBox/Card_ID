import cv2
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image

def load_vietocr():
    """
    Load mô hình VietOCR.
    """
    print("[INFO] Loading VietOCR model...")
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu'  # Đổi thành 'cuda' nếu có GPU
    return Predictor(config)

def extract_text_from_boxes(image, boxes, detector):
    """
    Nhận diện văn bản từ các vùng text crop được.
    """
    texts = {}
    for label, (x1, y1, x2, y2) in boxes.items():
        cropped_image = image[y1:y2, x1:x2]
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image_pil = Image.fromarray(cropped_image_rgb)

        text = detector.predict(cropped_image_pil)
        texts[label] = text.strip()  # Loại bỏ khoảng trắng thừa

        print(f"[INFO] Extracted {label}: {text}")

    return texts
