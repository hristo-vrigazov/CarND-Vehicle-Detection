from car_not_car_classifier import CarNotCarClassifier
from utils import find_cars
from moviepy.editor import VideoFileClip

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

    def detect_in_image(self, image):
        svc = self.car_not_car_classifier.svc
        X_scaler = self.car_not_car_classifier.X_scaler
        return find_cars(image, svc, X_scaler)

    def detect_in_video(self, input_file_name, output_file_name):
        input_clip = VideoFileClip(input_file_name)
        output_clip = input_clip.fl_image(self.detect_in_image)
        output_clip.write_videofile(output_file_name, audio=False)
