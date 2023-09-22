from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load('xgb_model.pkl')

# Define a request body model
class Customer(BaseModel):
    # Define features here, for example:
    Age: float
    Income: float
    # Add other features as needed

@app.post('/predict')
async def predict(customer: Customer):
    # Extract features from request
    features = [customer.Age, customer.Income]  # Update based on actual features
    prediction = model.predict([features])
    return {'prediction': prediction[0]}