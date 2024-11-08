def predict_churn(customer_data):
    # Load the model
    model = joblib.load('models/')

    # Preprocess customer_data here (e.g., scaling, encoding)
    customer_data = scaler.transform(customer_data)

    # Predict churn
    return model.predict(customer_data)
