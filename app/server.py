from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
import numpy as np
from pathlib import Path

__version__ ="0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/final_model-{__version__}.pkl", "rb") as f:
     model = pickle.load(f)


class_names = ['Arrival', 'Departure', 'Empty Container Released', 'Empty Return',
       'Gate In', 'Gate Out', 'In-transit', 'Inbound Terminal',
       'Loaded on Vessel', 'Off Rail', 'On Rail', 'Outbound Terminal',
       'Port In', 'Port Out', 'Unloaded on Vessel']

def predict_pipeline(text):
        text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9/]', " ", text)
        text = re.sub(r"[[]]", " ", text)
        text = text.lower()
        pred = model.predict([text])
        return class_names[pred[0]]

class TextIn(BaseModel):
    External_Status : str

class PredictionOut(BaseModel):
    Internal_Status: str    

app = FastAPI()

#@app.get("/")
#def read_root():
    #return{'message ': 'Cargo Status API'}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    #features = np. array(data['features']).reshape(1, -1)
    prediction = predict_pipeline(payload.External_Status)
    return{'Internal_Status' : prediction}