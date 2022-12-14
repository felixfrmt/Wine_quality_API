import sys
import os
import joblib
import numpy as np
import pandas as pd
sys.path.append('./AI models/')

from random_forest import Random_Forest
from preprocessing import Preprocessing as ppc


from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
 
from pydantic import BaseModel, conlist, Field
from typing import Union, List, Dict, Optional


class Wine_without_quality(BaseModel):
    fixed_acidity : float = 0
    volatile_acidity : float = 0
    citric_acid  : float = 0
    residual_sugar  : float = 0
    chlorides  : float = 0
    free_sulfur_dioxide  : float = 0
    total_sulfur_dioxide  : float = 0
    density : float = 0
    pH : float = 0
    sulphates : float = 0
    alcohol : float = 0


class Wine_and_quality(BaseModel):
    fixed_acidity : float = 0
    volatile_acidity : float = 0
    citric_acid  : float = 0
    residual_sugar  : float = 0
    chlorides  : float = 0
    free_sulfur_dioxide  : float = 0
    total_sulfur_dioxide  : float = 0
    density : float = 0
    pH : float = 0
    sulphates : float = 0
    alcohol : float = 0
    quality : float = 0


app = FastAPI(title="Wines ML/DL API", description="API for predictions and models about Wines dataset", version="1.0")

@app.on_event("startup")
async def startup_event():
	global df
	global X_train
	global X_test
	global y_train
	global y_test
	global RF_model
	df = ppc.load_data("./data/Wines.csv")
	X_train, X_test, y_train, y_test = ppc.split_data(df)
	RF_model = Random_Forest(X_train, y_train, X_test, y_test)


@app.post('/api/predict')
def predict_quality(wine:Wine_without_quality):
    x = pd.Series(wine.__dict__).to_frame().T
    y_pred = RF_model.test_model(x)
    return {"prediction": int(y_pred)}


@app.get('/api/predict')
def predict_quality():
    df.pop("Id")
    max_quality = max(df['quality'].to_numpy())
    max_quality = df[df['quality']==max_quality]
    max_quality.pop("quality")
    best_wine = pd.Series([0]*11, index=max_quality.columns)
    temp_wine = pd.Series([0]*11, index=max_quality.columns)
    for col in max_quality.columns:
        best_wine[col] = max_quality[col].iloc[0]
        temp_wine[col] = best_wine[col]
        
        for index, row in max_quality.iterrows():
            temp_wine[col] = row[col]
            
            y_pred_best_wine = RF_model.test_model(best_wine.to_frame().T)
            y_pred_temp_wine = RF_model.test_model(temp_wine.to_frame().T)

            if y_pred_temp_wine > y_pred_best_wine:
                best_wine = temp_wine.copy()

    return {"ideal_wine": best_wine}


@app.get('/api/model')
def get_serialized_model(model_name='model_random_forest.joblib'):
    path = os.path.join('./AI models/models/', model_name)
    
    if os.path.exists(path):
        return FileResponse(path, media_type="text/plain", filename=model_name)
    else:
    	return None


@app.get('/api/model/description')
def get_model_description():
    accuracy = pd.read_csv('./AI models/models/accuracy.csv')
    return {"number_estimators": len(RF_model.model.estimators_), "params": RF_model.model.get_params(), "accuracy": accuracy[accuracy["model"] == "Random_Forest"]["accuracy"]}



@app.put('/api/model')
def add_wine(wine: Wine_and_quality):
    df2 = ppc.load_data("./data/Wines2.csv")
    l = list(wine.__dict__.values())
    l.append(len(df2))
    df2.loc[len(df2)] = l
    df2.to_csv("./data/Wines2.csv", index=False)
    return {'put': 'wine was added'}


@app.post('/api/model/retrain')
def retrain_model(wine: Union[Wine_and_quality, None] = None):
    if wine is not None:
        add_wine(wine)

    df = ppc.load_data("./data/Wines.csv")
    X_train, X_test, y_train, y_test = ppc.split_data(df)
    
    RF_model.model = RF_model.train_model(X_train, y_train)
    RF_model.test_model(X_test, y_test)
    return {"post": "model retrained"}



if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
