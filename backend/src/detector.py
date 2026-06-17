from paddleocr import PaddleOCR

class TextDetector:
    def __init__(self):
        self.ocr = PaddleOCR(lang="en")

    def detect_boxes(self, img):
        # Gọi PaddleOCR quét ảnh để lấy tọa độ các hộp chữ
        results = self.ocr.ocr(img, det=True, rec=False)
        
        if not results or not results[0]:
            return []
        # Lấy tọa độ 4 góc của từng hộp chữ
        boxes = []
        for item in results[0]:
            if len(item) == 4:
                boxes.append(item)
        return boxes