import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data/wine_data_raw.csv')
X = df.drop('quality', axis=1)  # All columns except 'quality'
y = df['quality']  # Target variable is 'quality'

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test data
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
plt.title('Wine Quality Prediction - Random Forest')
plt.savefig('confusion_matrix.png')
plt.close()

# Print feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False))

