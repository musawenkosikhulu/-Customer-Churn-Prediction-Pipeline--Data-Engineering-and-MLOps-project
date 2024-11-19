import os
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import requests

# Load data from /data directory
data_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
data = pd.read_csv(data_path)

# Check for missing values
data_info = data.isnull().sum()
data.dropna(inplace=True)

# Label encode binary columns (e.g., 'gender')
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
                                      'MultipleLines', 'InternetService', 'OnlineSecurity', 
                                      'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                                      'StreamingTV', 'StreamingMovies', 'Contract', 
                                      'PaperlessBilling', 'PaymentMethod'], drop_first=True)

# Split features (X) and target (y)
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the model (RandomForestClassifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Ensure the /models directory exists
os.makedirs('models', exist_ok=True)

# Save the trained model to the models directory
model_path = 'models/churn_model.pkl'
joblib.dump(model, model_path)
joblib.dump(scaler, 'models/scaler.pkl')

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)  # Assuming '1' is the churn label
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

# Log the model evaluation results
log_message = f"Model trained successfully.\n\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"

# Send email notification with training results
def send_email(subject, body):
    sender_email = os.getenv("EMAIL_USERNAME")
    receiver_email = os.getenv("GITHUB_EMAIL")
    password = os.getenv("EMAIL_PASSWORD")
    
    # Create email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Send the email using SMTP
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()  # Encrypt the connection
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            print(f"Email sent to {receiver_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

send_email('Model Retraining Complete', log_message)

# Send Telegram notification
def send_telegram_message(message):
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': message}
    response = requests.post(url, data=payload)
    
    if response.status_code == 200:
        print("Telegram message sent successfully.")
    else:
        print(f"Failed to send Telegram message. Response code: {response.status_code}")

send_telegram_message(log_message)
