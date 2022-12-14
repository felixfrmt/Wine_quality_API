import pandas as pd
import joblib 
import os 

# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier



class Random_Forest:

	def __init__(self, X_train, y_train, X_test, y_test, model_name='model_random_forest.joblib'):
		path = os.path.join('./AI models/models/'+model_name)
		if os.path.exists(path):
			print("Model loaded")
			self.model = self.load_model(model_name) 
		else:
			print("Model created")
			self.model = self.train_model(X_train, y_train) 
			self.test_model(X_test, y_test)
			self.save_model(model_name)



	def load_model(self, model_name='model_random_forest.joblib'):
		return joblib.load('./AI models/models/'+model_name)

	def save_model(self, model_name='model_random_forest.joblib'):
		joblib.dump(self.model, './AI models/models/'+model_name)

	def train_model(self, X_train, y_train):
		print("Training model")
		self.model = RandomForestClassifier(n_estimators=150, oob_score=True, verbose=0)
		self.model.fit(X_train, y_train.to_numpy().flatten())
		return self.model

	def test_model(self, X_test, y_test=None):
		y_pred = self.model.predict(X_test)
		if y_test is not None:
			accuracy = self.model.score(X_test, y_test)
			df_acc = pd.DataFrame({"model": "Random_Forest", "accuracy":[accuracy]})
			
			path = os.path.join('./AI models/models/accuracy.csv')
			if os.path.exists(path):
				df_acc_file = pd.read_csv(path, sep=',')
				df_acc = pd.concat([df_acc_file, df_acc], axis=0)

			df_acc.to_csv(path, index=False)
		return y_pred

