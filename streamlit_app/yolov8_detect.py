from ultralytics import YOLO
import random
import cv2
def detect(image):
    model = YOLO("best_yolov8.pt")
    results = model.predict(image)
    result = results[0]
    status = 0
    box = []
    if len(result.boxes) > 0:
        for i in range(len(result.boxes)):
            box.append(result.boxes[i])
            class_id, xyxy, conf, xywh = box[i].cls.item() , box[i].xyxy.tolist(), box[i].conf.item(), box[i].xywh.tolist()
            # coordinates, class_id, conf = box[i].xyxy[0].tolist(), box[i].cls[0].item(), box[i].conf[0].item()
            x1, y1, x2, y2 = int(xyxy[0][0]), int(xyxy[0][1]), int(xyxy[0][2]), int(xyxy[0][3])
            class_name = result.names[class_id]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (random.randint(0,255), random.randint(0,255), random.randint(0,255)), 24)
            image = cv2.putText(image, class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            status = 1
    else:
        status = -1
    return status, image