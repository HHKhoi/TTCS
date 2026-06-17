import cv2

def preprocess_image(image_path, scale):
    img = cv2.imread(image_path)
    img_large = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img_large