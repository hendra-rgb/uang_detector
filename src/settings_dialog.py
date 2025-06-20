from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QDoubleSpinBox, QSpinBox, QPushButton, QFormLayout
)
import json
import os

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pengaturan Parameter")
        self.setMinimumWidth(300)

        self.config_path = "config.json"
        self.config = self.load_config()

        layout = QVBoxLayout()
        form = QFormLayout()

        self.canny1 = QSpinBox()
        self.canny1.setRange(0, 500)
        self.canny1.setValue(self.config.get("canny_threshold1", 50))

        self.canny2 = QSpinBox()
        self.canny2.setRange(0, 500)
        self.canny2.setValue(self.config.get("canny_threshold2", 150))

        self.blur = QSpinBox()
        self.blur.setRange(1, 31)
        self.blur.setSingleStep(2)
        self.blur.setValue(self.config.get("gaussian_blur", 5))

        self.entropy = QDoubleSpinBox()
        self.entropy.setRange(0.0, 10.0)
        self.entropy.setSingleStep(0.1)
        self.entropy.setValue(self.config.get("entropy_threshold", 3.0))

        form.addRow("Canny Threshold 1:", self.canny1)
        form.addRow("Canny Threshold 2:", self.canny2)
        form.addRow("Gaussian Blur:", self.blur)
        form.addRow("Entropy Threshold:", self.entropy)

        self.save_btn = QPushButton("Simpan")
        self.save_btn.clicked.connect(self.save_settings)

        layout.addLayout(form)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)

    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                return json.load(f)
        return {}

    def save_settings(self):
        self.config["canny_threshold1"] = self.canny1.value()
        self.config["canny_threshold2"] = self.canny2.value()
        self.config["gaussian_blur"] = self.blur.value()
        self.config["entropy_threshold"] = self.entropy.value()

        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=4)

        self.accept()
