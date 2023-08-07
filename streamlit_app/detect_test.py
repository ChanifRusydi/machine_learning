import cv2
from yolov8_detect import detect

image1 = cv2.imread('bus.jpg ')
status, image_detect = detect(image1)
cv2.imwrite('image_detect_result.jpg', image_detect)