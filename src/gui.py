from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget,
    QHBoxLayout, QGroupBox, QFileDialog, QMessageBox, QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import os

from src.camera_handler import CameraHandler
from src.processor import ImageProcessor
from src.settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🔍 Deteksi Keaslian Uang - Kelompok 5")
        self.setGeometry(150, 80, 1280, 720)
        self.setStyleSheet("background-color: #f0f0f0;")

        self.processor = ImageProcessor()
        self.camera_handler = CameraHandler()
        self.camera_handler.new_frame.connect(self.update_frame)
        self.camera_handler.start()

        self.init_ui()
        self.init_connections()

    def init_ui(self):
        # Kamera View
        self.original_view = QLabel("Tampilan Kamera")
        self.original_view.setAlignment(Qt.AlignCenter)
        self.original_view.setFrameShape(QFrame.Box)
        self.original_view.setMinimumSize(640, 480)
        self.original_view.setStyleSheet("background-color: #222; color: white;")

        # Processed View
        self.processed_view = QLabel("Hasil Deteksi")
        self.processed_view.setAlignment(Qt.AlignCenter)
        self.processed_view.setFrameShape(QFrame.Box)
        self.processed_view.setMinimumSize(640, 480)
        self.processed_view.setStyleSheet("background-color: #444; color: white;")

        # Hasil deteksi
        self.result_label = QLabel("Silakan buka gambar atau aktifkan kamera")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2d3436;")
        self.result_label.setFixedHeight(60)

        # Tombol kontrol
        self.capture_btn = QPushButton("📷 Ambil Gambar")
        self.load_btn = QPushButton("📂 Buka File")
        self.settings_btn = QPushButton("⚙️ Pengaturan")
        self.quit_btn = QPushButton("❌ Keluar")

        # Style tombol
        for btn in [self.capture_btn, self.load_btn, self.settings_btn, self.quit_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #0984e3;
                    color: white;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #74b9ff;
                }
            """)
            btn.setCursor(Qt.PointingHandCursor)

        # Layout tombol
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.capture_btn)
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.settings_btn)
        button_layout.addWidget(self.quit_btn)

        button_group = QGroupBox("Kontrol")
        button_group.setLayout(button_layout)

        # Layout kiri & kanan
        kiri = QVBoxLayout()
        kiri.addWidget(self.original_view)

        kanan = QVBoxLayout()
        kanan.addWidget(self.processed_view)
        kanan.addWidget(self.result_label)

        main_content = QHBoxLayout()
        main_content.addLayout(kiri)
        main_content.addLayout(kanan)

        # Gabungkan semua layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(main_content)
        main_layout.addWidget(button_group)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def init_connections(self):
        self.capture_btn.clicked.connect(self.capture_image)
        self.load_btn.clicked.connect(self.load_image)
        self.settings_btn.clicked.connect(self.show_settings)
        self.quit_btn.clicked.connect(self.close)

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.original_view.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.original_view.width(), self.original_view.height(), Qt.KeepAspectRatio
        ))

    def show_processed_image(self, image):
        if len(image.shape) == 2:
            h, w = image.shape
            qt_img = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.processed_view.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.processed_view.width(), self.processed_view.height(), Qt.KeepAspectRatio
        ))

    def capture_image(self):
        print("[DEBUG] Tombol Ambil Gambar diklik")
        frame = self.camera_handler.capture_image()
        if frame is not None:
            print("[DEBUG] Gambar berhasil ditangkap")
            self.process_image(frame)
        else:
            QMessageBox.warning(self, "Kamera", "Gagal menangkap gambar.")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Buka Gambar", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            print(f"[DEBUG] Gambar dimuat: {file_path}")
            image = cv2.imread(file_path)
            if image is not None:
                self.process_image(image)

    def process_image(self, image):
        self.update_frame(image)
        roi = self.processor.find_watermark_roi(image)
        processed = self.processor.preprocess_image(image, roi)
        edges = self.processor.detect_watermark(processed)
        nominal, confidence = self.processor.detect_nominal(image)
        result, edge_pixels, entropy = self.processor.analyze_watermark(edges)

        self.show_processed_image(edges)
        self.result_label.setText(
            f"📝 Hasil: <b>{result}</b> | Nominal: <b>{nominal}</b> ({confidence:.2f})<br>"
            f"🧮 Edge Pixels: {edge_pixels} | Entropy: {entropy:.2f}"
        )

    def show_settings(self):
        print("[DEBUG] Tombol Pengaturan diklik")
        dialog = SettingsDialog(parent=self)
        if dialog.exec_():
            QMessageBox.information(self, "Berhasil", "Pengaturan disimpan.")
    
    def closeEvent(self, event):
        if hasattr(self, 'camera_handler'):
            self.camera_handler.stop()
        event.accept()
