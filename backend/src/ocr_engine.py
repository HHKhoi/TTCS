import cv2
import os

# Fix crash on Windows CPU with PaddleOCR 3.x by disabling MKLDNN
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

from paddleocr import PaddleOCR

class OCREngine:
    def __init__(self):
        self.ocr = PaddleOCR(
            lang="en"
        )

    def extract_text(self, img_or_path):
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
            if img is None:
                raise ValueError(f"Cannot read image: {img_or_path}")
        else:
            img = img_or_path

        results = self.ocr.ocr(img)

        if not results or not results[0]:
            return ""

        texts = []
        
        # Support both PaddleOCR v3.x (returns dict) and v2.x (returns list)
        if isinstance(results[0], dict):
            rec_texts = results[0].get('rec_texts', [])
            if rec_texts:
                texts.extend(rec_texts)
        else:
            for line in results[0]:
                if len(line) >= 2:
                    box, (text, score) = line
                    texts.append(text)

        return " ".join(texts)