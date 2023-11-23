from ultralytics import YOLO
import cv2
import time
import random
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolov8n.pt")

# object classes
classNames =['person','bicycle','car','motorcycle','bus','train','truck','traffic light','fire hydrant','stop sign','cat','dog']

def detect(image):
    time_start = time.time_ns()
    results = model.predict(image)
    time_stop = time.time_ns()
    delta_time = (start_time - stop_time) / 1000
    fps = 1000/delta_time
    result = results[0]
    cv2.putText(image, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    status = 0
    box = []
    #TODO make multi thread for each deteted object 
    if len(result.boxes) > 0:
        for i in range(len(result.boxes)):
            box.append(result.boxes[i])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            class_id, xyxy, conf, xywh = box[i].cls.item() , box[i].xyxy.tolist(), box[i].conf.item(), box[i].xywh.tolist()
            # coordinates, class_id, conf = box[i].xyxy[0].tolist(), box[i].cls[0].item(), box[i].conf[0].item()
            x1, y1, x2, y2 = int(xyxy[0][0]), int(xyxy[0][1]), int(xyxy[0][2]), int(xyxy[0][3])
            class_name = result.names[class_id]
            image = cv2.rectangle(image, (x1, y1), (x2, y2),color=color, thickness=2)
            image = cv2.putText(image, class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color=color, thickness=2)
            status = 1
    else:
        status = -1
    return status, image
while True:
    success, image = cap.read()
    status, image = detect(image)
    cv2.imshow("Detect", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
# cv2.destroyAllWindows()