from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QSpinBox, QDialogButtonBox

class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pengaturan Deteksi")
        self.config = config

        layout = QVBoxLayout()

        self.threshold1_spin = QSpinBox()
        self.threshold1_spin.setRange(0, 500)
        self.threshold1_spin.setValue(config.get("canny_threshold1", 50))
        layout.addWidget(QLabel("Canny Threshold 1"))
        layout.addWidget(self.threshold1_spin)

        self.threshold2_spin = QSpinBox()
        self.threshold2_spin.setRange(0, 500)
        self.threshold2_spin.setValue(config.get("canny_threshold2", 150))
        layout.addWidget(QLabel("Canny Threshold 2"))
        layout.addWidget(self.threshold2_spin)

        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(1, 15)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setValue(config.get("gaussian_blur", 5))
        layout.addWidget(QLabel("Gaussian Blur (kernel size)"))
        layout.addWidget(self.blur_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_settings(self):
        return {
            "canny_threshold1": self.threshold1_spin.value(),
            "canny_threshold2": self.threshold2_spin.value(),
            "gaussian_blur": self.blur_spin.value()
        }
