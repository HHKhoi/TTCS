from paddleocr import PaddleOCR

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

    def extract_text(self, img):
        # Gọi PaddleOCR để nhận diện chữ trong ảnh 
        results = self.ocr.ocr(img)
        if not results or not results[0]:
            return ""
        texts = []
        for line in results[0]:
            text_info = line[1]      
            text = text_info[0]      
            texts.append(text)
        return " ".join(texts)
