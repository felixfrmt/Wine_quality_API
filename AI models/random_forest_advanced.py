import sys
 
# setting path
sys.path.append('../')
 
# importing
from preprocessing import Preprocessing as ppc

#from .. import preprocessing

import pandas as pd
from collections import Counter

# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import pickle for saving model
import pickle


def train_model(X_train, y_train):
	# Create a Gaussian Classifier
	model = RandomForestClassifier(n_estimators=200)
	# Train the model using the training sets y_pred=clf.predict(X_test)
	model.fit(X_train, y_train)
	return model


def test_model(model, X_test, y_test=[], message=""):
	y_pred = model.predict(X_test)
	# Model Accuracy, how often is the classifier correct?
	# if len(y_test) >= 0:
		# print(f"\n\n{message}")
		# print(f"\tAccuracy : {metrics.accuracy_score(y_test, y_pred)}\n\n")
	return y_pred


def load_model(path='model.sav'):
	with open(path , 'rb') as f:
		model = pickle.load(f)
	return model

def save_model(model, filename='model.sav'):
	with open(filename, 'wb') as files:
		pickle.dump(model, files)




df = ppc.load_data()
df = ppc.generate_group(df)

X_train, X_test, y_train, y_test = ppc.split_data(df)

y_group_train = y_train.pop("group")
y_quality_train = y_train.pop("quality")

y_group_test = y_test.pop("group")
y_quality_test = y_test.pop("quality")

# Modèle de prédiction pour le groupe
model_group = train_model(X_train, y_group_train)
y_group_pred = test_model(model_group, X_test, y_group_test, "Validité du modèle pour la selection de groupe : ")


index_1 = [index for index, y in y_group_train.items() if y == 1]
index_2 = [index for index, y in y_group_train.items() if y == 2]
index_3 = [index for index, y in y_group_train.items() if y == 3]

X_group_1_train = X_train.loc[index_1]
X_group_2_train = X_train.loc[index_2]
X_group_3_train = X_train.loc[index_3]
y_quality_group_1_train = y_quality_train.loc[index_1]
y_quality_group_2_train = y_quality_train.loc[index_2]
y_quality_group_3_train = y_quality_train.loc[index_3]


model_group_1 = train_model(X_group_1_train, y_quality_group_1_train)
model_group_2 = train_model(X_group_2_train, y_quality_group_2_train)
model_group_3 = train_model(X_group_3_train, y_quality_group_3_train)


index_1 = [index for index, y in y_group_test.items() if y == 1]
index_2 = [index for index, y in y_group_test.items() if y == 2]
index_3 = [index for index, y in y_group_test.items() if y == 3]

X_group_1_test = X_test.loc[index_1]
X_group_2_test = X_test.loc[index_2]
X_group_3_test = X_test.loc[index_3]
y_quality_group_1_test = y_quality_test.loc[index_1]
y_quality_group_2_test = y_quality_test.loc[index_2]
y_quality_group_3_test = y_quality_test.loc[index_3]

# Modèle de prédiction pour la qualité suivant le groupe 
y_quality_group_1_pred = test_model(model_group_1, X_group_1_test, y_quality_group_1_test, "Validité du modèle pour la prédiction de la qualité du groupe 1 : ")
y_quality_group_2_pred = test_model(model_group_2, X_group_2_test, y_quality_group_2_test, "Validité du modèle pour la prédiction de la qualité du groupe 2 : ")
y_quality_group_3_pred = test_model(model_group_3, X_group_3_test, y_quality_group_3_test, "Validité du modèle pour la prédiction de la qualité du groupe 3 : ")


# Test du modèle de prédiction général : prédiction du groupe, puis prédiction de la qualité.
y_group_pred = test_model(model_group, X_test)

print(X_test)
print(len(y_group_pred))

predictions = []
i = 0
for index, row in X_test.iterrows():
	
	y = y_group_pred[i]
	if y == 1:
		y_pred = test_model(model_group_1, [row])
	elif y == 2:
		y_pred = test_model(model_group_2, [row])
	else:
		y_pred = test_model(model_group_3, [row])
	
	i += 1
	predictions.append(int(y_pred))

general_accuracy = metrics.accuracy_score(y_quality_test, predictions)


print(f"\nAccuracy general : \t{general_accuracy}")

output = X_test.copy()
output.insert(len(output.columns), "group", y_group_test)
output.insert(len(output.columns), "group_prediction", y_group_pred)
output.insert(len(output.columns), "quality", y_quality_test)
output.insert(len(output.columns), "prediiction", predictions)

counter_quality = Counter(df["quality"])
counter_group = Counter(df["group"])
print(f"\nNombre de vins pour chaque qualité : \t{counter_quality}")
print(f"\nNombre de vins pour chaque group : \t{counter_group}\n")

print(f"Nombre de tests effectués : {len(X_test)}\n")

difference = [q - p for q, p in zip(y_quality_test, predictions)]
counter_difference = Counter(difference)

print(f"Nombre d'erreurs : {len([d for d in difference if d != 0])}\n")
print(f"\nNombre d'erreurs suivant la différence entre la qualité réelle et la prédiction : \t{counter_difference}\n")


output.insert(len(output.columns), "difference", difference)

erreur_group_1 = [row["difference"] for index, row in output.iterrows() if row["difference"] != 0 and row["group"] == 1]
erreur_group_2 = [row["difference"] for index, row in output.iterrows() if row["difference"] != 0 and row["group"] == 2]
erreur_group_3 = [row["difference"] for index, row in output.iterrows() if row["difference"] != 0 and row["group"] == 3]

nb_erreur_group_1 = len(erreur_group_1)
nb_erreur_group_2 = len(erreur_group_2)
nb_erreur_group_3 = len(erreur_group_3)

quantity_erreur_group_1 = sum(erreur_group_1)
quantity_erreur_group_2 = sum(erreur_group_2)
quantity_erreur_group_3 = sum(erreur_group_3)

counter_difference_group_1 = Counter(erreur_group_1)
counter_difference_group_2 = Counter(erreur_group_2)
counter_difference_group_3 = Counter(erreur_group_3)

print(f"Nombre d'erreurs présentes dans le groupe 1 : {nb_erreur_group_1}\nSomme des erreurs : {quantity_erreur_group_1}\n{counter_difference_group_1}\n")
print(f"Nombre d'erreurs présentes dans le groupe 2 : {nb_erreur_group_2}\nSomme des erreurs : {quantity_erreur_group_2}\n{counter_difference_group_2}\n")
print(f"Nombre d'erreurs présentes dans le groupe 3 : {nb_erreur_group_3}\nSomme des erreurs : {quantity_erreur_group_3}\n{counter_difference_group_3}\n")


output.to_csv("predictions.csv", index=False)
