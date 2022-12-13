import pandas as pd
import joblib 

# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics



class Random_Forest:


	def __init__(self, X_train, y_train, X_test, y_test):
		self.model = self.train_model(X_train, y_train)
		self.y_pred = self.test_model(X_test)
		self.accuracy = metrics.accuracy_score(y_test, self.y_pred)
		self.save_model()


	def load_model(self, model_name='model_random_forest.joblib'):
		return joblib.load('./AI models/models/'+model_name)

	def save_model(self, model_name='model_random_forest.joblib'):
		joblib.dump(self.model, './AI models/models/'+model_name)

	def train_model(self, X_train, y_train):
		# Create a Gaussian Classifier
		self.model = RandomForestClassifier(n_estimators=150, verbose=1)
		# Train the model using the training sets y_pred=clf.predict(X_test)
		self.model.fit(X_train, y_train.to_numpy().flatten())
		print("Colonnes caractéristiques : \n", self.model.feature_importances_)
		return self.model

	def test_model(self, X_test):
		y_pred = self.model.predict(X_test)			
		return y_pred



