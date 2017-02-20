from imutils import paths
from utils import extract_features
from utils import save_array, load_array

import numpy as np
import matplotlib.image as mpimg
import os

class CarNotCarClassifier:

	def __init__(self, 
		serialized_data_folder='./',
		path_to_vehicle_folder='vehicles/', 
		path_to_non_vehicle_folder='non-vehicles/'):

		path_to_X_dat_file = '{0}/{1}'.format(serialized_data_folder, 'X.dat')
		path_to_y_dat_file = '{0}/{1}'.format(serialized_data_folder, 'y.dat')
		X_dat_file_exists = os.path.exists(path_to_X_dat_file)
		y_dat_file_exists = os.path.exists(path_to_y_dat_file)

		if X_dat_file_exists and y_dat_file_exists:
			print('Loading from serialized ...')
			self.X = load_array(path_to_X_dat_file)
			self.y = load_array(path_to_y_dat_file)
			print('Done reading serialized arrays')
		else:
			print('Creating data from image folders')
			non_vehicle_class, vehicle_class = 0, 1
			non_vehicle_X, non_vehicle_y = self._get_X_y(path_to_non_vehicle_folder, non_vehicle_class)
			vehicle_X, vehicle_y = self._get_X_y(path_to_vehicle_folder, vehicle_class)

			self.X = np.concatenate((non_vehicle_X, vehicle_X))
			self.y = np.concatenate((non_vehicle_y, vehicle_y))
			print('Data created successfully, creating serialized numpy arrays')
			save_array(path_to_X_dat_file, self.X)
			save_array(path_to_y_dat_file, self.y)
			print('Done saving arrays')

		

	def _get_X_y(self, path_to_images, class_label):
		X = extract_features(paths.list_images(path_to_images))
		y = np.full(len(X), class_label)
		return X, y

