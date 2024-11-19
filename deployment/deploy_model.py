import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

def predict_churn(customer_data):
    # Load the trained model
    model = joblib.load('models/churn_model.pkl')  # Ensure you're loading the correct model file
    
    # Load the scaler used during training
    scaler = joblib.load('models/scaler.pkl')  # Assuming you saved the scaler as 'scaler.pkl'
    
    # Example: Assuming customer_data is a pandas DataFrame
    customer_data_scaled = scaler.transform(customer_data)  # Apply the same scaling as the training data

    # Predict churn
    return model.predict(customer_data_scaled)  # Predict using the model
