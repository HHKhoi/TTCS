from paddleocr import PaddleOCR
from src.preprocess import preprocess_image
from src.detector import TextDetector
from src.box_utils import sort_boxes, crop_box
import numpy as np

# Đường dẫn mô hình đã fine-tune
MODEL_FINETUNE = "D:/AI-OCR-Detection/backend/train_rec/output/rec_ppocrv4_finetuned_infer"

class OCREngine:
    def __init__(self):
        self.ocr = PaddleOCR(
            lang="en",
            rec_model_dir=MODEL_FINETUNE, 
            use_gpu=False,
            show_log=False
        )

    def recognize_crop(self, crop):
        results = self.ocr.ocr(crop, det=False, rec=True)
        if not results or not results[0]:
            return ""
        text = results[0][0][0]
        return text

    def extract_text(self, img_path):
        

        detector = TextDetector()

        # Phóng to ảnh lên 2 lần 
        img = preprocess_image(img_path, 2)

        # Tìm các khung chữ trong ảnh 
        boxes = detector.detect_boxes(img)
        if not boxes:
            return ""
            
        # Sắp xếp các hộp chữ theo thứ tự từ trên xuống dưới
        boxes = sort_boxes(boxes)
        
        lines = []
        # Cắt từng khung chữ và mang đi nhận diện 
        for box in boxes:
            crop = crop_box(img, box, pad=2)
            if crop is None or crop.size == 0:
                continue
            text = self.recognize_crop(crop)
            if text.strip():
                lines.append(text.strip())
        
        # 5. Gộp các dòng chữ bằng dấu xuống dòng
        return "\n".join(lines)


