import os
from dotenv import load_dotenv, find_dotenv

# Load .env trước khi kiểm tra
load_dotenv(find_dotenv('.env'), override=True)

# Liệt kê danh sách các biến cần kiểm tra
required_vars = [
    "FRONTEND_USERNAME", "FRONTEND_PASSWORD",
    "FACE_ALIGN__FILE_CONFIG_PATH",
    "CARD_DETECTOR__MODEL_PATH", "CARD_DETECTOR__CONF",
    "CARD_ALIGN__BASEIMG_PATH", "CARD_ALIGN__PER_MATCH",
    "TEXT_DETECTOR__MODEL_PATH", "TEXT_DETECTOR__CONF",
    "TEXT_OCR__CONFIG_NAME", "TEXT_OCR__DEVICE", "TEXT_OCR__MODEL_PATH",
    "TEXT_OCR__CNN_PRETRAINED", "TEXT_OCR__VOCAB",
    "HOST_CARD_DETECTOR", "HOST_TEXT_DETECTOR", "HOST_TEXT_OCR",
    "HOST_OCR_SERVICE", "BASE_IMG",
    "POSTGRES__USERNAME", "POSTGRES__PASSWORD", "POSTGRES__HOST", "POSTGRES__DB"
]

# Kiểm tra từng biến môi trường
missing_vars = [var for var in required_vars if os.getenv(var) is None]

# In ra các biến thiếu
if missing_vars:
    print("Các biến môi trường thiếu:", missing_vars)
    print("\nDanh sách các biến thiếu:")
    for var in missing_vars:
        print(var)
else:
    print("Tất cả các biến môi trường đã được tải đúng.")
