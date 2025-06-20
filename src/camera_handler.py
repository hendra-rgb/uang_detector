import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

class CameraHandler(QThread):
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.camera = None
        self.running = False

    def run(self):
        self.camera = cv2.VideoCapture(self.camera_index)
        self.running = True
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                self.new_frame.emit(frame)

    def stop(self):
        self.running = False
        self.wait()
        if self.camera:
            self.camera.release()

    def capture_image(self):
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                return frame
        return None
