import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget,
    QHBoxLayout, QGroupBox, QFileDialog, QMessageBox, QFrame
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from src.camera_handler import CameraHandler
from src.processor import ImageProcessor
from src.settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deteksi Keaslian Uang - Kelompok 5")
        self.setGeometry(100, 100, 1200, 800)

        self.processor = ImageProcessor()
        self.camera_handler = CameraHandler()
        self.camera_handler.new_frame.connect(self.update_frame)
        self.camera_handler.start()

        self.init_ui()
        self.init_connections()

    def init_ui(self):
        self.original_view = QLabel(self)
        self.original_view.setAlignment(Qt.AlignCenter)
        self.original_view.setMinimumSize(640, 480)
        self.original_view.setFrameShape(QFrame.Box)
        self.original_view.setStyleSheet("border: 2px solid black;")

        self.processed_view = QLabel(self)
        self.processed_view.setAlignment(Qt.AlignCenter)
        self.processed_view.setMinimumSize(640, 480)
        self.processed_view.setFrameShape(QFrame.Box)
        self.processed_view.setStyleSheet("border: 2px solid black;")

        self.result_label = QLabel("Arahkan kamera ke uang kertas dengan pencahayaan dari belakang", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.capture_btn = QPushButton("Ambil Gambar", self)
        self.load_btn = QPushButton("Buka File", self)
        self.settings_btn = QPushButton("Pengaturan", self)
        self.quit_btn = QPushButton("Keluar", self)

        control_panel = QGroupBox("Kontrol")
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.capture_btn)
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.settings_btn)
        control_layout.addWidget(self.quit_btn)
        control_panel.setLayout(control_layout)

        view_panel = QGroupBox("Tampilan")
        view_layout = QHBoxLayout()
        view_layout.addWidget(self.original_view)
        view_layout.addWidget(self.processed_view)
        view_panel.setLayout(view_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(view_panel)
        main_layout.addWidget(self.result_label)
        main_layout.addWidget(control_panel)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def init_connections(self):
        self.capture_btn.clicked.connect(self.capture_image)
        self.load_btn.clicked.connect(self.load_image)
        self.settings_btn.clicked.connect(self.show_settings)
        self.quit_btn.clicked.connect(self.close)

    def update_frame(self, frame):
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
        if len(image.shape) == 2:
            h, w = image.shape
            qt_image = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.processed_view.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.processed_view.width(),
            self.processed_view.height(),
            Qt.KeepAspectRatio
        ))

    def capture_image(self):
        print("[DEBUG] Tombol Ambil Gambar diklik")
        frame = self.camera_handler.capture_image()
        if frame is not None:
            print("[DEBUG] Gambar berhasil ditangkap dari kamera")
            self.process_image(frame)
        else:
            print("[DEBUG] Gagal menangkap gambar dari kamera")
            QMessageBox.warning(self, "Kamera", "Gagal menangkap gambar dari kamera.")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Buka Gambar Uang", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

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
        result, edge_pixels, entropy = self.processor.analyze_watermark(edges)
        self.show_processed_image(edges)
        self.result_label.setText(
            f"Hasil: {result}\nEdge Pixels: {edge_pixels}\nEntropy: {entropy:.2f}"
        )

    def show_settings(self):
        print("[DEBUG] Tombol Pengaturan diklik")
        dialog = SettingsDialog(self.processor.config, self)
        if dialog.exec_() == dialog.Accepted:
            new_config = dialog.get_settings()
            self.processor.config.update(new_config)
            self.processor.save_config()
            QMessageBox.information(self, "Pengaturan", "Pengaturan berhasil disimpan.")

    def closeEvent(self, event):
        if hasattr(self, 'camera_handler'):
            self.camera_handler.stop()
        event.accept()
