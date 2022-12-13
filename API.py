import sys
import os
import numpy as np
import pandas as pd
# setting path
sys.path.append('./AI models/')

from Random_Forest import Random_Forest
from preprocessing import Preprocessing as ppc

import joblib
import json

from typing import Union, List, Dict, Optional
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
 
from pydantic import BaseModel, conlist, Field

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


# @app.on_event("startup")
# async def startup_event():
#     df = ppc.load_data("./data/Wines.csv")
#     X_train, X_test, y_train, y_test = ppc.split_data(df)
#     RF_model = Random_Forest(X_train, y_train, X_test, y_test)

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
    mean_features = max_quality.mean(axis=0)
    return {"mean_of_best": mean_features}


@app.get('/api/model')
def get_serialized_model(model_name='model_random_forest.joblib'):
    path = os.path.join('./AI models/models/', model_name)
    
    if os.path.exists(path):
        return FileResponse(path, media_type="text/plain", filename=model_name)


@app.get('/api/model/description')
def get_model_description():
    return {"number_estimators": len(RF_model.model.estimators_), "params": RF_model.model.get_params(), "accuracy": RF_model.accuracy}



@app.put('/api/model')
def add_wine(wine: Wine_and_quality):
    df = ppc.load_data("./data/Wines.csv")
    l = list(wine.__dict__.values())
    l.append(len(df))
    df.loc[len(df)] = l
    df.to_csv("./data/Wines2.csv", index=False)
    return {'put': 'wine was added'}


@app.post('/api/model/retrain')
def retrain_model(wine: Union[Wine_and_quality, None] = None):
    if wine is not None:
        add_wine(wine)

    df = ppc.load_data("./data/Wines.csv")
    X_train, X_test, y_train, y_test = ppc.split_data(df)
    RF_model = Random_Forest(X_train, y_train, X_test, y_test)
    return {"post": "retraining done"}



if __name__ == '__main__':
    df = ppc.load_data("./data/Wines.csv")
    X_train, X_test, y_train, y_test = ppc.split_data(df)
    RF_model = Random_Forest(X_train, y_train, X_test, y_test)


    uvicorn.run(app, host="127.0.0.1", port=8000)
