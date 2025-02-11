import os

# Lấy đường dẫn tuyệt đối đến thư mục gốc của dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Định nghĩa đường dẫn model
MODELS_DIR = os.path.join(BASE_DIR, "models")

CORNER_MODEL_PATH = os.path.join(MODELS_DIR, "card_detect.pt")
TEXT_MODEL_PATH = os.path.join(MODELS_DIR, "text_detection.pt")
