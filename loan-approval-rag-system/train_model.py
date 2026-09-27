import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

print("Loading dataset...")
# 1. Load the CSV you downloaded
df = pd.read_csv('data/loan_dataset.csv')

# 2. Clean data (fill missing values)
df['Credit_History'] = df['Credit_History'].fillna(1.0)
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())

# 3. Select features to train on
X = df[['Credit_History', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']]
y = df['Loan_Status'].map({'Y': 1, 'N': 0}) # Convert Yes/No to 1/0

print("Training Machine Learning Model...")
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 4. Save the trained model to the models/ folder
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/loan_model.pkl')
print("✅ Success! 'loan_model.pkl' has been created in the models folder.")