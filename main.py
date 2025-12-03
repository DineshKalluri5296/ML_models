from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("USA_Housing.pkl")

class USA_HousingInput(BaseModel):
   Avg. Area Income = float
   Avg. Area House Age = float
   Avg. Area Number of Rooms = float
   Avg. Area Number of Bedrooms = float
   Area Population = float


@app.get("/")
def read_root():
    return {"message": "USA_Housing Prediction API is live"}

@app.post("/predict")
def predict(data: USA_HousingsInput):
    input_data = np.array([[data.Avg. Area Income, data.Avg. Area House Age, data.Avg. Area Number of Rooms, data.Avg. Area Number of BedRooms, data.Area Population]])
    prediction = model.predict(input_data)
    return {"USA_Housing": bool(prediction)}
