import cv2
import numpy as np
import json
import os

class ImageProcessor:
    def __init__(self, config_path='config.json'):
        self.load_config(config_path)
        self.templates = {
            "10000": "samples/asli/10000.jpg",
            "20000": "samples/asli/20000.jpg",
            "50000": "samples/asli/50000.jpg",
            "100000": "samples/asli/100000.jpg"
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
                "histogram_tile_size": 8,
                "entropy_threshold": 3.0
            }

    def preprocess_image(self, image, roi=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]

        clahe = cv2.createCLAHE(
            clipLimit=self.config['histogram_clip_limit'],
            tileGridSize=(
                self.config['histogram_tile_size'],
                self.config['histogram_tile_size']
            )
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

        if (edge_pixels > self.config['watermark_threshold']) and (entropy > self.config["entropy_threshold"]):
            return "Asli", edge_pixels, entropy
        else:
            return "Palsu", edge_pixels, entropy

    def find_watermark_roi(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            return (x, y, w, h)

        return None

    def detect_nominal(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        best_match = None
        best_score = 0

        for nominal, path in self.templates.items():
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is None:
                print(f"[WARNING] Template tidak ditemukan: {path}")
                continue

            if gray.shape[0] < template.shape[0] or gray.shape[1] < template.shape[1]:
                print(f"[WARNING] Ukuran gambar terlalu kecil untuk template {nominal}")
                continue

            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            print(f"[DEBUG] Pencocokan {nominal}: Skor = {max_val:.3f}")

            if max_val > best_score:
                best_score = max_val
                best_match = nominal

        return best_match or "Tidak dikenali", best_score
