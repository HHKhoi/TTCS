import cv2
import numpy as np
from paddleocr import PaddleOCR

# Create a dummy image
img = np.zeros((100, 300, 3), dtype=np.uint8)
cv2.putText(img, 'Hello World', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

try:
    ocr = PaddleOCR(lang="en", use_angle_cls=False)
    # try with no kwargs
    res1 = ocr.ocr(img)
    print("No kwargs SUCCESS")
except Exception as e:
    print(f"No kwargs ERROR: {e}")

try:
    # try with det, rec
    res2 = ocr.ocr(img, det=True, rec=False)
    print("det=True, rec=False SUCCESS")
except Exception as e:
    print(f"det=True, rec=False ERROR: {e}")
