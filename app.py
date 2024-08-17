from fastapi import FastAPI
import uvicorn
import numpy as np
import pandas as pd
import pickle
import json
from pydantic import BaseModel

app = FastAPI()

class inputvar(BaseModel):
    age : float
    sex : float
    bmi : float
    bp : float
    s1 : float
    s2 : float
    s3 : float
    s4 : float
    s5 : float
    s6 : float
    
    
    
    
# Chargement du modele
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaling.pkl','rb'))

@app.get('/')
def index():
    return {'message':'Hello Henri BOUITYVOUBOU'}

@app.post('/predict')
def prediction(Data: inputvar):
    data = pd.json_normalize(Data.dict())
    data = scalar.transform(data)
    
    predicted = regmodel.predict(data)
    
    return f"The diabete progression prediction is {predicted}"
    
    

if __name__ == '__main__':
    uvicorn.run(app)