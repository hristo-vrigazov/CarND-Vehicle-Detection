from car_not_car_classifier import CarNotCarClassifier
from utils import slide_window

import cv2
import numpy as np

from skimage.transform import resize

class CarDetector:
    def __init__(self, car_not_car_classifier=None):
        if car_not_car_classifier:
            self.car_not_car_classifier = car_not_car_classifier
        else:
            self.car_not_car_classifier = CarNotCarClassifier()
            self.car_not_car_classifier.load_data()
            self.car_not_car_classifier.train()

    def detect_in_image(self, image, color=(0, 255, 0), thick=6):
        y_start_stop = [image.shape[0] // 2, image.shape[0]]
        image_copy = np.copy(image)
        for x in range(64, 129, 16):
            windows = slide_window(image, y_start_stop=y_start_stop, xy_window=(x, x))
            for window in windows:
                (top, left), (bottom, right) = window
                img = image[left:right, top:bottom]
                resized = resize(img, (64, 64))
                if self.car_not_car_classifier.predict_image(resized) == 1:
                    print('There is a car!')
                    cv2.rectangle(image_copy, (top, left), (bottom, right), color, thick)
        return image_copy
