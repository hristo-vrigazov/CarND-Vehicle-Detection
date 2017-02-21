from car_not_car_classifier import CarNotCarClassifier
from utils import slide_window

import cv2

class CarDetector:
    def __init__(self, car_not_car_classifier=None):
        if car_not_car_classifier:
            self.car_not_car_classifier = car_not_car_classifier
        else:
            self.car_not_car_classifier = CarNotCarClassifier()
            self.car_not_car_classifier.load_data()
            self.car_not_car_classifier.train()

    def detect_in_image(self, image, thick=6):
        windows = slide_window(image)
        for window in windows:
            (top, left), (bottom, right) = window
            img = image[left:right, top:bottom]
            if self.car_not_car_classifier.predict_image(img) == 1:
                print('There is a car!')
                cv2.rectangle(image, (top, left), (bottom, right), thick)
        return image
