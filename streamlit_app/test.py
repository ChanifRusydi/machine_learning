from ultralytics import YOLO
import random
import cv2

camera1 = cv2.VideoCapture(0)
camera2 = cv2.VideoCapture(1)

while True:
    ret1, frame1 = camera1.read()
    # ret2, frame2 = camera2.read()
    # Display the resulting frame
    cv2.imshow('frame1', frame1)
    # cv2.imshow('frame2', frame2)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
camera1.release()
camera2.release()
cv2.destroyAllWindows()

# def detect(image):
#     model = YOLO("best_yolov8.pt")
#     results = model.predict(source=image)
#     result = results[0]
#     status = 0
#     box = []
#     if len(result.boxes) > 0:
#         for i in range(len(result.boxes)):
#             box.append(result.boxes[i])
#             class_id, xyxy, conf, xywh = box[i].cls.item() , box[i].xyxy.tolist(), box[i].conf.item(), box[i].xywh.tolist()
#             # coordinates, class_id, conf = box[i].xyxy[0].tolist(), box[i].cls[0].item(), box[i].conf[0].item()
#             x1, y1, x2, y2 = int(xyxy[0][0]), int(xyxy[0][1]), int(xyxy[0][2]), int(xyxy[0][3])
#             class_name = result.names[class_id]
#             image = cv2.rectangle(image, (x1, y1), (x2, y2), (random.randint(0,255), random.randint(0,255), random.randint(0,255)), 2)
#             image = cv2.putText(image, class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             status = 1
#     else:
#         status = -1
#     return status, image
# image = cv2.imread('bus.jpg')
# status_code, image = detect(image)
# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()