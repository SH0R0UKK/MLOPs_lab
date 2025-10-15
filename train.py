import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load data
df = pd.read_csv('data/train_data.csv')

X_train = df.drop('quality', axis=1) # All columns except 'quality'
y_train = df['quality'] # Target variable is 'quality'

# Train model
model = LogisticRegression(random_state=42, max_iter=2000)
model.fit(X_train, y_train)

# Save the trained model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.joblib')
print("Model training complete and saved to 'models/model.joblib'")