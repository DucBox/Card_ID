import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Lấy thư mục hiện tại của src/
MODELS_DIR = os.path.join(BASE_DIR, "models")

CORNER_MODEL_PATH = os.path.join(MODELS_DIR, "card_detect.pt")
TEXT_MODEL_PATH = os.path.join(MODELS_DIR, "text_detection.pt")

# Debug path
print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
print(f"[DEBUG] CORNER_MODEL_PATH: {CORNER_MODEL_PATH}")
print(f"[DEBUG] TEXT_MODEL_PATH: {TEXT_MODEL_PATH}")
