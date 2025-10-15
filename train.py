import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/wine_data_raw.csv')
X = df.drop('quality', axis=1)
y = df['quality']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
preds = model.predict(X_test_scaled)

# Save metrics
acc = accuracy_score(y_test, preds)
with open('metrics.json', 'w') as f:
    json.dump({'accuracy': acc}, f)

# Generate confusion matrix plot
cm = confusion_matrix(y_test, preds, labels=model.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted Quality')
plt.ylabel('Actual Quality')
plt.title('Wine Quality Prediction - SVM')
plt.savefig('confusion_matrix.png')
plt.close()

