import cv2
import numpy as np

IMG_SIZE = (224, 224)

def preprocess_image(path):
    img = cv2.imread(path)

    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img