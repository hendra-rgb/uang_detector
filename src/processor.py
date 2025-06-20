import cv2
import numpy as np
import json
import os
from skimage import exposure, filters
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class ImageProcessor:
    def __init__(self, config_path='config.json'):
        self.load_config(config_path)
        self.model_path = "model_ai.pkl"
        self.model = self.load_ai_model()
        self.label_map = {
            10000: 0,
            20000: 1,
            50000: 2,
            100000: 3
        }

    def load_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "canny_threshold1": 50,
                "canny_threshold2": 150,
                "gaussian_blur": 5,
                "watermark_threshold": 1000,
                "histogram_clip_limit": 2.0,
                "histogram_tile_size": 8
            }

    def save_config(self, path='config.json'):
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)

    def preprocess_image(self, image, roi=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]
        clahe = cv2.createCLAHE(
            clipLimit=self.config['histogram_clip_limit'],
            tileGridSize=(self.config['histogram_tile_size'], self.config['histogram_tile_size'])
        )
        equalized = clahe.apply(gray)
        return equalized

    def detect_watermark(self, image):
        blurred = cv2.GaussianBlur(
            image,
            (self.config['gaussian_blur'], self.config['gaussian_blur']),
            0
        )
        edges = cv2.Canny(
            blurred,
            self.config['canny_threshold1'],
            self.config['canny_threshold2']
        )
        return edges

    def analyze_watermark(self, edges):
        edge_pixels = np.sum(edges > 0)
        hist = cv2.calcHist([edges], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        if edge_pixels > self.config['watermark_threshold'] and entropy > 3.0:
            return "Asli", edge_pixels, entropy
        else:
            return "Palsu", edge_pixels, entropy

    def find_watermark_roi(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            return (x, y, w, h)
        return None

    def extract_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi = self.find_watermark_roi(image)
        if roi:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        histogram = cv2.calcHist([clahe], [0], None, [32], [0, 256])
        histogram = cv2.normalize(histogram, histogram).flatten()
        return histogram

    def train_ai_model(self, samples_dir="samples/asli"):
        data = []
        labels = []
        for filename in os.listdir(samples_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                label_str = filename.split("_")[0]
                try:
                    nominal = int(label_str)
                    img_path = os.path.join(samples_dir, filename)
                    image = cv2.imread(img_path)
                    if image is not None:
                        features = self.extract_features(image)
                        data.append(features)
                        labels.append(self.label_map.get(nominal, -1))
                except ValueError:
                    continue
        data = np.array(data)
        labels = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[INFO] Akurasi Model: {acc:.2f}")
        joblib.dump(model, self.model_path)
        self.model = model

    def load_ai_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        else:
            print("[WARNING] Model AI tidak ditemukan.")
            return None

    def detect_nominal(self, image):
        if self.model is None:
            return "Model belum tersedia", 0.0
        features = self.extract_features(image).reshape(1, -1)
        pred = self.model.predict(features)[0]
        conf = np.max(self.model.predict_proba(features)) if hasattr(self.model, "predict_proba") else 1.0
        reverse_label_map = {v: k for k, v in self.label_map.items()}
        nominal = reverse_label_map.get(pred, "Unknown")
        return nominal, conf
