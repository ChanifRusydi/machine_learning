from ultralytics import YOLO
import random
def detect(image):
    model = YOLO("best_yolov8.pt")
    results = model.predict(image)
    result = results[0]
    status = 0
    if len(result.boxes) > 0:
        for i in range(len(result.boxes)):
            box[i]= result.boxes[i]
            cls, xyxy, conf = box[i].cls , box[i].xyxy, box[i].conf
            coordinates, class_id, conf = box[i].xyxy[0].tolist(), box[i].cls[0].item(), box[i].conf[0].item()
            x1, y1, x2, y2 = coordinates
            
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (random.randint(0,255) random.randint(0,255), random.randint(0,255)), 2)
            image = cv2.putText(image, result.names[i], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            status = 1
    else:
        status = -1
    return status, image