import cv2
from paddleocr import PaddleOCR


class OCREngine:
    def __init__(self):
        self.ocr = PaddleOCR(
            lang="en",
            rec_model_dir="backend/train_rec/output/rec_ppocrv4_finetuned_infer",
            use_angle_cls=False,
            show_log=False
        )

    def extract_text(self, img_or_path):
        if isinstance(img_or_path, str):
            img = cv2.imread(img_or_path)
            if img is None:
                raise ValueError(f"Cannot read image: {img_or_path}")
        else:
            img = img_or_path
        results = self.ocr.ocr(img)
        if not results:
            return ""
        texts = []
        for line in results:
            if not line:
                continue
            for item in line:
                if item is None or len(item) < 2:
                    continue
                text = item[1][0]
                texts.append(text)
        return "\n".join(texts)