import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

import supervision as sv


def detect_objects():

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    assert camera.isOpened()
    
    model = YOLO('yolov8n.pt')
    model.fuse()
    box_annotator = sv.BoxAnnotator(thickness=2)


    while True:
        ret, frame = camera.read()
        # cv2.imshow('frame', frame)
        results = model(frame)
        print(results)
        # for detection in results[0]:
        #     class_id = detection.cls
        #     confidence = detection.conf
        #     xyxy = detection.xyxy
        #     if class_id == 0.0:
        #         print(f"{class_id} {confidence:0.2f}")
        #         print(xyxy)
        #     print(class_id, confidence, xyxy)
        # detections = sv.Detections.from_yolov8(results[0].boxes)
        # frame= box_annotator.annotate(scene=frame, detections=detections)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    detect_objects()