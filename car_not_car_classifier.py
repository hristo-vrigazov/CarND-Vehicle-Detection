from imutils import paths
from utils import extract_features

import numpy as np
import matplotlib.image as mpimg

class CarNotCarClassifier:

	def __init__(self, path_to_vehicle_folder='vehicles/', path_to_non_vehicle_folder='non-vehicles/'):
		non_vehicle_class, vehicle_class = 0, 1
		non_vehicle_X, non_vehicle_y = self._get_X_y(path_to_non_vehicle_folder, 0)
		vehicle_X, vehicle_y = self._get_X_y(path_to_vehicle_folder, 1)

		X = np.concatenate((non_vehicle_X, vehicle_X))
		y = np.concatenate((non_vehicle_y, vehicle_y))

		print(len(X))
		print(len(y))
		

	def _get_X_y(self, path_to_images, class_label):
		X = extract_features(paths.list_images(path_to_images))
		y = np.full(len(X), class_label)
		return X, y

