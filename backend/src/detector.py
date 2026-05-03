from paddleocr import PaddleOCR
from src.settings import DET_LANG

class TextDetector:
    def __init__(self):
        self.ocr = PaddleOCR(
            lang=DET_LANG,
            use_angle_cls=False,
            use_gpu=False,         
            enable_mkldnn=False,    
            show_log=False,
        )

    def detect_boxes(self, img):
        results = self.ocr.ocr(img, det=True, rec=False, cls=False)
        boxes = []
        if not results or not results[0]:
            return boxes
        for item in results[0]:
            if isinstance(item, list) and len(item) == 4:
                boxes.append(item)
            elif (
                isinstance(item, list)
                and len(item) > 0
                and isinstance(item[0], list)
                and len(item[0]) == 4
            ):
                boxes.append(item[0])
        return boxes