from imutils import paths
from utils import extract_features
from utils import save_array, load_array
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import numpy as np
import matplotlib.image as mpimg
import os
import time


class CarNotCarClassifier:

	def load_data(self, 
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


	def train(self):
		X_scaler = StandardScaler().fit(self.X)
		scaled_X = X_scaler.transform(self.X)
		X_train, X_test, y_train, y_test = train_test_split(
			scaled_X, self.y, test_size=0.2, random_state=7)

		svc = LinearSVC()
		print('Start fitting')
		start_time = time.time()
		svc.fit(X_train, y_train)
		print('Time for training: ', time.time() - start_time, ' seconds')
		print('Train Accuracy of SVC = ', svc.score(X_train, y_train))
		print('Test Accuracy of SVC = ', svc.score(X_test, y_test))

	def _get_X_y(self, path_to_images, class_label):
		X = np.array(extract_features(paths.list_images(path_to_images))).astype(np.float64)
		y = np.full(len(X), class_label)
		return X, y