import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load diabetes dataset from sklearn
from sklearn.datasets import load_diabetes
data = load_diabetes(as_frame=True)
X = data.data
y = (data.target > data.target.mean()).astype(int)  # Convert regression target to 0/1 classification

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save trained model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as diabetes_model.pkl")