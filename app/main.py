from fastapi import FastAPI
import joblib
import pandas as pd
import os
from pydantic import BaseModel

# =========================
# INPUT SCHEMA
# =========================
class CarInput(BaseModel):
    Model: str
    Fuel_Type: str
    Turbo: str
    Horsepower: float

# =========================
# INIT APP
# =========================
app = FastAPI()

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

model = joblib.load(MODEL_PATH)

# =========================
# ROOT ENDPOINT
# =========================
@app.get("/")
def home():
    return {"status": "API is running 🚀"}

# =========================
# PREDICT ENDPOINT
# =========================
@app.post("/predict")
def predict(data: CarInput):
    try:
        df = pd.DataFrame([{
            "Model": data.Model,
            "Fuel Type": data.Fuel_Type,  # mapping correct
            "Turbo": data.Turbo,
            "Horsepower": data.Horsepower
        }])

        prediction = model.predict(df)[0]

        return {"prediction": float(prediction)}

    except Exception as e:
        return {"error": str(e)}