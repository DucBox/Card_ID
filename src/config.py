import os

# Đường dẫn đến model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORNER_MODEL_PATH = os.path.join(BASE_DIR, "../models/card_detect.pt")
TEXT_MODEL_PATH = os.path.join(BASE_DIR, "../models/text_recog.pt")
