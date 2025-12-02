from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("kyphosis.pkl")

class KyphosisInput(BaseModel):
    Number:int
    Start:int
    Age: int

@app.get("/")
def read_root():
    return {"message": "Kyphosis Prediction API is live"}

@app.post("/predict")
def predict(data: KyphosisInput):
    input_data = np.array([[data.Number, data.Age, data.Start]])
    prediction = model.predict(input_data)[0]
    return {"Kyphosis": bool(prediction)}
