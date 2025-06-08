import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load Olivetti Faces dataset
faces = fetch_olivetti_faces()
X = faces.data  # 4096 features (64x64 images flattened)
y = faces.target  # Labels from 0 to 39 (person ID)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Gaussian Naive Bayes Classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict on test data
y_pred = gnb.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy of Gaussian Naive Bayes on Olivetti Faces:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))