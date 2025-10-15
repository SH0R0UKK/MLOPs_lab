from sklearn.metrics import accuracy_score, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Load data
df = pd.read_csv('data/test_data.csv')
X_test = df.drop('quality', axis=1)  # All columns except 'quality'
y_test = df['quality']  # Target variable is 'quality'

# Load the trained model
model = joblib.load('models/model.joblib')

# Make predictions on test set
preds = model.predict(X_test)

# Save metrics
acc = accuracy_score(y_test, preds)
with open('metrics.json', 'w') as f:
    json.dump({'accuracy': acc}, f)

# Generate and save confusion matrix plot
cm = confusion_matrix(y_test, preds, labels=model.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Quality')
plt.ylabel('Actual Quality')
plt.title('Wine Quality Prediction Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()
