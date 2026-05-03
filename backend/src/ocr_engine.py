import cv2
from paddleocr import PaddleOCR


class OCREngine:
    def __init__(self):
        self.ocr = PaddleOCR(
            lang="en",
            use_angle_cls=False,
            use_gpu=False,
            show_log=False,
        )

    def extract_text(self, img_or_path):
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
            if img is None:
                raise ValueError(f"Cannot read image: {img_or_path}")
        else:
            img = img_or_path
        results = self.ocr.ocr(img, det=False, rec=True, cls=False)
        texts = []

        def walk(x):
            if isinstance(x, str):
                texts.append(x)
                return
            if isinstance(x, tuple):
                x = list(x)
            if isinstance(x, list):
                if len(x) >= 2 and isinstance(x[0], str) and isinstance(x[1], (int, float)):
                    texts.append(x[0])
                    return
                for t in x:
                    walk(t)
        walk(results)
        return " ".join(t.strip() for t in texts if t and t.strip())