from PySide6.QtCore import *
from PySide6.QtGui import *
# from PySide6.QtWidgets import *
from PySide6 import QtWidgets
from PySide6.QtGui import QWindow

import cv2
import sys

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # self.video_size = QSize(720,480)
        self.setup_ui()
        self.setup_camera()

    def setup_ui(self):
        """Initialize widgets.
        """
        self.image_label = QtWidgets.QLabel()
        # self.image_label.setFixedSize(self.video_size)

        # self.quit_button = QtWidgets.QPushButton("Quit")
        # self.quit_button.clicked.connect(self.close)

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        # self.main_layout.addWidget(self.quit_button)

        self.setLayout(self.main_layout)

    def setup_camera(self):
        """Initialize camera.
        """
        self.capture = cv2.VideoCapture(1)
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)

    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget.
        """
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        image = QImage(frame, frame.shape[1], frame.shape[0], 
                       frame.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MyWidget()
    window.showFullScreen()
    sys.exit(app.exec())