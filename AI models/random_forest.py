import sys
 
# setting path
sys.path.append('../')

import preprocessing as ppc

import os
import pandas as pd

# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import pickle for saving model
import pickle


def train_model(X_train, y_train):
	# Create a Gaussian Classifier
	model = RandomForestClassifier(n_estimators=150, verbose=1)
	# Train the model using the training sets y_pred=clf.predict(X_test)
	model.fit(X_train, y_train)
	return model


def test_model(model, X_test, y_test):
	y_pred = model.predict(X_test)
	# Model Accuracy, how often is the classifier correct?
	print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
	return y_pred


def load_model(path='model.sav'):
	with open(path , 'rb') as f:
		model = pickle.load(f)
	return model


def choose_model(X_train, y_train, path="./", model_name="./model.sav"):
	files = []
	model_format = model_name[-4:]
	print(model_format)
	for r, d, f in os.walk(path):
		for file in f:
			if file.endswith(model_format):
				files.append(os.path.join(r, file))
	print(files)
	if model_name in files:
		print("Model loaded")
		model = load_model(model_name)
	else:
		print("Model created")
		model = train_model(X_train, y_train)
	return model

def save_model(model, filename='model.sav'):
	with open(filename, 'wb') as files:
		pickle.dump(model, files)



df = ppc.load_data("../Wines.csv")
df = ppc.generate_group(df)

X_train, X_test, y_train, y_test = ppc.split_data(df)

model = choose_model(X_train, y_train["group"])

y_pred = test_model(model, X_test, y_test["group"])

# save_model(model)