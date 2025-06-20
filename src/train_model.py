import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 64))
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, channel_axis=None)
    return features

def load_data():
    X, y = [], []
    for label, folder in enumerate(['asli', 'palsu']):
        path = os.path.join('samples', folder)
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            image = cv2.imread(img_path)
            if image is not None:
                features = extract_features(image)
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)

# Load dan latih model
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\n[INFO] Evaluasi Model:")
print(classification_report(y_test, model.predict(X_test)))

# Simpan model
joblib.dump(model, "model_uang.pkl")
print("[INFO] Model berhasil disimpan: model_uang.pkl")
