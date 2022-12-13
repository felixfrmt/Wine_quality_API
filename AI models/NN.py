import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys
# setting path
sys.path.append('../')

from preprocessing import Preprocessing as ppc



class NN:

	def __init__(self, X_train, y_train, X_test, y_test, load_model_name="model.sav"):
		
		if load_model_name :
			try:
				print("Model loaded")
				self.model = keras.models.load_model("./models_NN/" + load_model_name)
			except:
				print("Model created")
				NN.__init__(self, X_train, y_train, load_model_name=None)
		else:
			print("Model created")

			# Train the model for 1 epoch from Numpy data
			batch_size = 64

			val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

			# Build a simple model
			inputs = keras.Input(shape=(11))
			x = layers.Normalization(axis=-1)(inputs)
			x = layers.Dense(11, activation="relu")(x)
			x = layers.Dropout(0.1)(x)
			x = layers.Dense(8, activation="relu")(x)
			x = layers.Dropout(0.7)(x)
			outputs = layers.Dense(10, activation="softmax")(x)

			self.model = keras.Model(inputs, outputs)
			self.model.summary()

			# Compile the model
			self.model.compile(
				optimizer="adam", 
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])
			
			self.model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))


	def save_model(self, filename='model'):
		self.model.save('./models_NN/' + filename)


df = ppc.load_data()
X_train, X_test, y_train, y_test = ppc.split_data(df, y="quality")


model = NN(X_train, y_train, X_test, y_test, None)
model.save_model()

