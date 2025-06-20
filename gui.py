from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QVBoxLayout, QWidget, QHBoxLayout, QGroupBox,
                            QFileDialog, QSpinBox, QDoubleSpinBox, QComboBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2
import numpy as np
import json
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deteksi Keaslian Uang - Kelompok 5")
        self.setGeometry(100, 100, 1200, 800)
        
        self.init_ui()
        self.init_connections()
        
    def init_ui(self):
        """Inisialisasi antarmuka pengguna"""
        # Widget utama
        self.original_view = QLabel(self)
        self.original_view.setAlignment(Qt.AlignCenter)
        self.original_view.setMinimumSize(640, 480)
        
        self.processed_view = QLabel(self)
        self.processed_view.setAlignment(Qt.AlignCenter)
        self.processed_view.setMinimumSize(640, 480)
        
        self.result_label = QLabel("Arahkan kamera ke uang kertas dengan pencahayaan dari belakang", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        # Tombol kontrol
        self.capture_btn = QPushButton("Ambil Gambar", self)
        self.load_btn = QPushButton("Buka File", self)
        self.settings_btn = QPushButton("Pengaturan", self)
        self.quit_btn = QPushButton("Keluar", self)
        
        # Panel kontrol
        control_panel = QGroupBox("Kontrol")
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.capture_btn)
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.settings_btn)
        control_layout.addWidget(self.quit_btn)
        control_panel.setLayout(control_layout)
        
        # Panel tampilan
        view_panel = QGroupBox("Tampilan")
        view_layout = QHBoxLayout()
        view_layout.addWidget(self.original_view)
        view_layout.addWidget(self.processed_view)
        view_panel.setLayout(view_layout)
        
        # Layout utama
        main_layout = QVBoxLayout()
        main_layout.addWidget(view_panel)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(control_panel)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
    def init_connections(self):
        """Hubungkan sinyal dan slot"""
        self.capture_btn.clicked.connect(self.capture_image)
        self.load_btn.clicked.connect(self.load_image)
        self.settings_btn.clicked.connect(self.show_settings)
        self.quit_btn.clicked.connect(self.close)
        
    def update_frame(self, frame):
        """Update tampilan frame asli"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.original_view.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.original_view.width(), 
            self.original_view.height(),
            Qt.KeepAspectRatio
        ))
        
    def show_processed_image(self, image):
        """Tampilkan gambar yang telah diproses"""
        if len(image.shape) == 2:  # Grayscale
            h, w = image.shape
            qt_image = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:  # Color
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
        self.processed_view.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.processed_view.width(),
            self.processed_view.height(),
            Qt.KeepAspectRatio
        ))
        
    def capture_image(self):
        """Tangkap gambar dari kamera"""
        # Implementasi akan ditambahkan setelah integrasi dengan camera_handler
        pass
        
    def load_image(self):
        """Muat gambar dari file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Buka Gambar Uang", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.process_image(image)
                
    def process_image(self, image):
        """Proses gambar untuk deteksi watermark"""
        # 1. Tampilkan gambar asli
        self.update_frame(image)
        
        # 2. Temukan ROI watermark
        roi = self.processor.find_watermark_roi(image)
        
        # 3. Preprocessing
        processed = self.processor.preprocess_image(image, roi)
        
        # 4. Deteksi watermark
        edges = self.processor.detect_watermark(processed)
        
        # 5. Analisis
        result, edge_pixels, entropy = self.processor.analyze_watermark(edges)
        
        # 6. Tampilkan hasil
        self.show_processed_image(edges)
        self.result_label.setText(
            f"Hasil: {result}\n"
            f"Edge Pixels: {edge_pixels}\n"
            f"Entropy: {entropy:.2f}"
        )
        
    def show_settings(self):
        """Tampilkan dialog pengaturan"""
        # Implementasi dialog pengaturan
        pass
        
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop kamera dan bersihkan resource
        if hasattr(self, 'camera_handler'):
            self.camera_handler.stop()
        event.accept()