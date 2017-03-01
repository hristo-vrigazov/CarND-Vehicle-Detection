from keras.models import model_from_json
from utils import draw_boxes

import cv2
import numpy as np


class YoloObjectDetector:
    def __init__(self, path_to_architecture='yolo_model.json', path_to_weights='yolo_weights.h5'):
        print('Reading architecture')
        with open(path_to_architecture) as json_file:
            json_string = json_file.read()
            self.model = model_from_json(json_string)
        print('Done. Loading weights')
        self.model.load_weights(path_to_weights)
        self.model.summary()
        print('Done loading weights')

        self.class_names_dict = {}
        self.class_names_dict['aeroplane'] = 0
        self.class_names_dict['bicycle'] = 1
        self.class_names_dict['bird'] = 2
        self.class_names_dict['boat'] = 3
        self.class_names_dict['bottle'] = 4
        self.class_names_dict['bus'] = 5
        self.class_names_dict['car'] = 6
        self.class_names_dict['cat'] = 7
        self.class_names_dict['chair'] = 8
        self.class_names_dict['cow'] = 9
        self.class_names_dict['diningtable'] = 10
        self.class_names_dict['dog'] = 11
        self.class_names_dict['horse'] = 12
        self.class_names_dict['motorbike'] = 13
        self.class_names_dict['person'] = 14
        self.class_names_dict['pottedplant'] = 15
        self.class_names_dict['sheep'] = 16
        self.class_names_dict['sofa'] = 17
        self.class_names_dict['train'] = 18
        self.class_names_dict['tvmonitor'] = 19

    def detect_in_cropped_image(self, image, class_name, x_start=500, x_stop=1280, y_start=300, y_stop=650):
        image_cropped = image[y_start:y_stop, x_start:x_stop, :]
        return self.detect_in_image(class_name, image.shape[0], image.shape[1], image_cropped)

    def detect_in_image(self, class_name, height, width, image_cropped):
        resized = cv2.resize(image_cropped, (448, 448))
        batch = np.transpose(resized, (2, 0, 1))
        batch = 2 * (batch / 255.) - 1
        batch = np.expand_dims(batch, axis=0)
        out = self.model.predict(batch)
        classes_to_bounding_boxes = self.extract_boxes_from_yolo_output(out[0])
        class_index = self.class_names_dict[class_name]
        car_bounding_boxes = classes_to_bounding_boxes[class_index]
        return list(self.convert_boxes(car_bounding_boxes, height, width))

    def extract_boxes_from_yolo_output(self, output_vector, threshold=0.2,
                                       sqrt=1.8,
                                       number_of_classes=20,
                                       number_of_boxes=2, S=7):
        # result is map from class to bounding boxes
        result = [[] for i in range(number_of_classes)]
        SS = S * S
        probability_sector_size = SS * number_of_classes
        confidence_sector_size = SS * number_of_boxes

        probabilities = output_vector[0: probability_sector_size]
        confidence = output_vector[probability_sector_size: (probability_sector_size + confidence_sector_size)]
        cords = output_vector[(probability_sector_size + confidence_sector_size):]
        probabilities = probabilities.reshape([SS, number_of_classes])
        confidence = confidence.reshape([SS, number_of_boxes])
        cords = cords.reshape([SS, number_of_boxes, 4])

        for grid in range(SS):
            for b in range(number_of_boxes):
                box = {}
                box['confidence'] = confidence[grid, b]
                box['x'] = (cords[grid, b, 0] + grid % S) / S
                box['y'] = (cords[grid, b, 1] + grid // S) / S
                box['width'] = cords[grid, b, 2] ** sqrt
                box['height'] = cords[grid, b, 3] ** sqrt
                p = probabilities[grid, :] * box['confidence']

                for class_num in range(number_of_classes):
                    if p[class_num] >= threshold:
                        box['probability'] = p[class_num]
                        result[class_num].append(box)

        return result

    def convert_boxes(self, boxes, height, width, x_start=500, x_stop=1280, y_start=300, y_stop=650):
        for box in boxes:
            left = int((box['x'] - box['width'] / 2.) * width)
            right = int((box['x'] + box['width'] / 2.) * width)
            top = int((box['y'] - box['height'] / 2.) * height)
            bot = int((box['y'] + box['height'] / 2.) * height)
            left = int(left * (x_stop - x_start) / width + x_start)
            right = int(right * (x_stop - x_start) / width + x_start)
            top = int(top * (y_stop - y_start) / height + y_start)
            bot = int(bot * (y_stop - y_start) / height + y_start)

            left = max(0, left)
            right = min(width - 1, right)
            top = max(0, top)
            top = max(0, top)
            bot = min(height - 1, bot)
            yield (left, top), (right, bot)