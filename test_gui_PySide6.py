from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6 import QtWidgets

import cv2
import sys
import logging

logging.basicConfig(filename='logfile.txt',filemode='a',format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)
logging.info('Start PYthon GUI App')
logger = logging.getLogger(__name__)
screen_width = QtWidgets.QDesktopWidget().screenGeometry().width()
screen_height = QtWidgets.QDesktopWidget().screenGeometry().height()
logging.info('Screen Size: %s x %s', screen_width, screen_height)

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

#capture two cameras and display them using different threads
# class Thread(QThread):
#     updateFrame = Signal(QImage)

#     def __init__(self, camera_id, parent=None):
#         QThread.__init__(self, parent)
#         self.trained_model = None
#         self.status = True
#         self.cap = True
#         self.camera_id = camera_id
#         self.camera = cv2.VideoCapture(camera_id)

#     def run(self):
#         while True:
#             ret, frame = self.camera.read()
#             if ret:
#                 h, w, ch = frame.shape
#                 img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
#                 pixmap = QPixmap.fromImage(img)
#                 scaled_img = pixmap.scaled(640, 480, Qt.KeepAspectRatio)
#                 self.updateFrame.emit(scaled_img)

# class Window(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setup_ui()
#         self.setup_camera()
        
#     def setup_ui(self):
#         self.setWindowTitle('My Window')
#         self.setGeometry(0, 0, 800, 600)
#         self.label1 = QLabel(self)
#         self.label1.setGeometry(50, 50, 320, 240)
#         self.label2 = QLabel(self)
#         self.label2.setGeometry(400, 50, 320, 240)

#     def setup_camera(self):
#         self.thread1 = Thread(0)
#         self.thread1.updateFrame.connect(self.label1.setPixmap)
#         self.thread1.start()

#         self.thread2 = Thread(1)
#         self.thread2.updateFrame.connect(self.label2.setPixmap)
#         self.thread2.start()
    
# if __name__ == "__main__":
#     app = QApplication([])
#     window = Window()
#     window.showFullScreen()
#     sys.exit(app.exec())