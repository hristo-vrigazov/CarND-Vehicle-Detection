import numpy as np
from moviepy.editor import VideoFileClip

from car_not_car_classifier import CarNotCarClassifier
from utils import draw_boxes, get_cars, draw_labeled_bounding_boxes
from utils import find_cars_bounding_boxes, create_heatmap

import matplotlib.pyplot as plt


class CarDetector:
    def __init__(self, car_not_car_classifier=None):
        if car_not_car_classifier:
            self.car_not_car_classifier = car_not_car_classifier
        else:
            self.car_not_car_classifier = CarNotCarClassifier()
            self.car_not_car_classifier.load_data()
            self.car_not_car_classifier.train()

    def detect_in_image(self, image):
        bounding_boxes = self._collect_bounding_boxes(image)
        img_size = image.shape[0], image.shape[1]
        heatmap = create_heatmap(img_size, bounding_boxes)
        labels = get_cars(heatmap)
        result = draw_labeled_bounding_boxes(image, labels)
        return result

    def _collect_bounding_boxes(self, image):
        svc = self.car_not_car_classifier.svc
        X_scaler = self.car_not_car_classifier.X_scaler
        bounding_boxes = []
        for scale in np.linspace(1, 2, num=4):
            bounding_boxes += find_cars_bounding_boxes(image, svc, X_scaler, scale=scale)
        return bounding_boxes

    def detect_in_video(self, input_file_name, output_file_name):
        input_clip = VideoFileClip(input_file_name)
        output_clip = input_clip.fl_image(self.detect_in_image)
        output_clip.write_videofile(output_file_name, audio=False)
