import cv2
import os
from yolov8_detect import detect
image_list= ['pagi.png', 'siang.png','sore.png','malam.png']
for image in image_list:
    image1 = cv2.imread(image)
    status, image_detect = detect(image1)
    file_name = os.path.basename(image) + '_detect.png'
    cv2.imwrite(filename=file_name,img=image_detect)