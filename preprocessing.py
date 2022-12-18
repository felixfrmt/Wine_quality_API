import os
import numpy as np
import pandas as pd

# Import train_test_split function
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt


class Preprocessing:

	def load_data(path='./data/Wines.csv', extension='csv', plot=False):
		print(path) 
		if extension == "xlsx":
			df = pd.read_excel(path, header=0)
		elif extension == "csv":
			df = pd.read_csv(path, sep=',', header=0)
		else:
			exit()
		df = df.astype('float')
		df = df.fillna(0)
		if plot:
			correlation = df.corr()
			sns.heatmap(correlation, 
				xticklabels=correlation.columns.values,
				yticklabels=correlation.columns.values)
			plt.show()
		
		return df

	def generate_group(df):
		group = []
		for index, row in df.iterrows():
			if row["quality"] <= 3.4:	
				group.append(1)
			elif 3.4 < row["quality"] and row["quality"] <= 6.8:	
				group.append(2)
			else :	
				group.append(3)
		
		df.insert(len(df.columns), "group", group)
		df.to_csv("../Wines_group.csv", index=False)
		return df


	def split_data(df, y=["quality"], size_train=0.3):
		X = df[["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol"]]  # Features
		y = df[y]

		# Split dataset into training set and test set
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size_train, shuffle=True) # 70% training and 30% test

		return X_train, X_test, y_train, y_test


	def oversampling():
		pass