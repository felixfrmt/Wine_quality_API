import sys
 
# setting path
sys.path.append('../')

from preprocessing import Preprocessing as ppc

import os
import pandas as pd

# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import pickle for saving model
import pickle


class Random_Forest:

	def __init__(self, X_train, y_train, load_model_name=None):
		if load_model_name :
			self.model = self.choose_model(X_train, y_train, path="./models_random_forest/", model_name=load_model_name)
		else:
			self.model = self.train_model(X_train, y_train)
			save_model(self.model, "retrained_model.sav")	



	def load_model(self, path='./models_random_forest/model.sav'):
		with open(path , 'rb') as f:
			self.model = pickle.load(f)
		return self.model

	def train_model(self, X_train, y_train):
		# Create a Gaussian Classifier
		self.model = RandomForestClassifier(n_estimators=150, verbose=1)
		# Train the model using the training sets y_pred=clf.predict(X_test)
		self.model.fit(X_train, y_train)
		return self.model

	def choose_model(self, X_train, y_train, path="./models_random_forest/", model_name="model.sav"):
		files = []
		model_format = model_name[-4:]
		# print(model_format)
		for r, d, f in os.walk(path):
			for file in f:
				if file.endswith(model_format):
					files.append(os.path.join(r, file))
		print(files)
		model_name = path + model_name 
		if model_name in files:
			print("Model loaded")
			self.model = self.load_model(model_name)
		else:
			print("Model created")
			self.model = self.train_model(X_train, y_train)
		return self.model

	def test_model(self, X_test, y_test):
		y_pred = self.model.predict(X_test)
		# Model Accuracy, how often is the classifier correct?
		print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
		return y_pred

	def save_model(self, filename='model.sav'):
		path = "./models_random_forest/" + filename
		with open(path, 'wb') as files:
			pickle.dump(self.model, files)





df = ppc.load_data()

X_train, X_test, y_train, y_test = ppc.split_data(df)

model = Random_Forest(X_train, y_train, load_model_name="model.sav")

# model = choose_model(X_train, y_train)

y_pred = model.test_model(X_test, y_test)

model.save_model()