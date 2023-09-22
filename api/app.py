from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = FastAPI()

# Initialize the model and scaler (you might want to load a pre-trained model instead)
scaler = StandardScaler()
rf_classifier = RandomForestClassifier(random_state=42)

class CustomerData(BaseModel):
    # Define the fields from the dataset here
    # For example:
    Age: float
    Gender: str
    # ... add other fields
    
class ClusterData(BaseModel):
    data: list[CustomerData]

@app.post("/predict_churn/")
def predict_churn(customer: CustomerData):
    try:
        # Convert the input data to DataFrame
        df = pd.DataFrame([dict(customer)])
        
        # Process and encode the data similar to how you did before model training
        # For example:
        df_encoded = pd.get_dummies(df, drop_first=True)
        
        # Scale the features
        X = scaler.transform(df_encoded)
        
        # Predict using the trained model
        prediction = rf_classifier.predict(X)
        
        return {"prediction": int(prediction[0])}
    except:
        raise HTTPException(status_code=400, detail="Error in prediction")

@app.post("/cluster_analysis/")
def cluster_analysis(cluster_data: ClusterData):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([dict(cd) for cd in cluster_data.data])
        
        # Process and encode the data similar to how you did before clustering
        # For example:
        df_encoded = pd.get_dummies(df, drop_first=True)
        
        # Scale the features
        X = scaler.transform(df_encoded)
        
        # Predict the cluster using KMeans
        cluster = kmeans.predict(X)
        
        # Get the characteristics of the cluster (you might want to store the characteristics beforehand)
        characteristics = cluster_analysis.loc[cluster].to_dict()
        
        return {"cluster": int(cluster[0]), "characteristics": characteristics}
    except:
        raise HTTPException(status_code=400, detail="Error in clustering")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
