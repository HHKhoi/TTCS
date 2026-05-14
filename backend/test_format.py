import cv2
import numpy as np
from paddleocr import PaddleOCR

# Create a dummy image with text
img = np.zeros((100, 300, 3), dtype=np.uint8)
cv2.putText(img, 'Test Text', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

ocr = PaddleOCR(lang="en")
results = ocr.ocr(img)

print("Raw results:", results)

if not results or not results[0]:
    print("No results found.")
else:
    for line in results[0]:
        print("Line:", line)
        try:
            box, (text, score) = line
            print("Successfully unpacked:", text, score)
        except Exception as e:
            print("Failed to unpack:", e)
