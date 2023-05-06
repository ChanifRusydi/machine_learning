import cv2
import argparse
import time
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-dual_camera", default=False, help="Use dual camera")


#  # if dual camera is True, then we are stitching the images from two cameras
# if args["dual_camera"]:
#     # capture the images from the cameras
#     camera1 = cv2.VideoCapture(0)
#     camera2 = cv2.VideoCapture(1)
#     # grab the frames from the cameraas
#     (grabbed1, frame1) = camera1.read()
#     (grabbed2, frame2) = camera2.read()
#     # stitch the frames together to form the panorama
#     stitch= Stitcher()
#     (result, vis) = stitch.stitch([frame1, frame2], showMatches=True)
#     # show the images
#     cv2.imshow("Keypoint Matches", vis)
#     cv2.imshow("Result", result)
#     # run object detection on the image
    
#     cv2.waitKey(0)
# else:
#     # run object detection on the image
#     image = cv2.imread("images/3.jpg")
#     image = imutils.resize(image, width=400)

#list available camera
# import cv2
# def returnCameraIndexes():
#     # checks the first 10 indexes.
#     index = 0
#     arr = []
#     i = 5
#     while i > 0:
#         cap = cv2.VideoCapture(index)
#         if cap.read()[0]:
#             arr.append(index)
#             cap.release()
#         index += 1
#         i -= 1
#         print(arr)
#         time.sleep(1)
#     return arr
# returnCameraIndexes()
# define a video capture obect
# camera1 = cv2.VideoCapture(2)
# camera2 = cv2.VideoCapture(4)
  
# while(True):
      
#     # Capture the video frame
#     # by frame
#     ret1, frame1 = camera1.read()
#     ret2, frame2 = camera2.read()

  
    # Display the resulting frame
#     cv2.imshow('frame1', frame2)
#     cv2.imshow('frame2', frame2)
      
#     # the 'q' button is set as the
#     # quitting button you may use any
#     # desired button of your choice
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
  
# # After the loop release the cap object
# camera1.release()
# camera2.release()
# # Destroy all the windows
# cv2.destroyAllWindows()


# import cv2
# import threading

# class camThread(threading.Thread):
#     def __init__(self, previewName, camID):
#         threading.Thread.__init__(self)
#         self.previewName = previewName
#         self.camID = camID
#     def run(self):
#         print("Starting " + self.previewName)
#         camPreview(self.previewName, self.camID)

# def camPreview(previewName, camID):
#     cv2.namedWindow(previewName)
#     cam = cv2.VideoCapture(camID)
#     if cam.isOpened():  # try to get the first frame
#         rval, frame = cam.read()
#     else:
#         rval = False

#     while rval:
#         cv2.imshow(previewName, frame)
#         # grab the frame using cam.grab() and then retrieve it using cam.retrieve()
#         grab = cam.grab()
#         print(typeof(grab), grab)
        
#         key = cv2.waitKey(20)
#         if key == 27:  # exit on ESC
#             break
#     cv2.destroyWindow(previewName)

# # Create two threads as follows
# thread1 = camThread("Camera 1", 2)
# thread2 = camThread("Camera 2", 4)
# thread1.start()
# thread2.start()

 
# import torch
# import numpy as np
# import cv2
# from time import time
# from ultralytics import YOLO

# import supervision as sv


# class ObjectDetection:

#     def __init__(self, capture_index):
       
#         self.capture_index = capture_index
        
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         print("Using Device: ", self.device)
        
#         self.model = self.load_model()
        
#         self.CLASS_NAMES_DICT = self.model.model.names
    
#         self.box_annotator = sv.BoxAnnotator(sv.ColorPalette.default(), thickness=3, text_thickness=3, text_scale=1.5)
    

#     def load_model(self):
       
#         # model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8n model
#         model = YOLO("yolov8n.pt")
#         model.fuse()
    
#         return model


#     def predict(self, frame):
       
#         results = self.model(frame)
        
#         return results
    

#     def plot_bboxes(self, results, frame):
        
#         xyxys = []
#         confidences = []
#         class_ids = []
        
#          # Extract detections for person class
#         for result in results:
#             boxes = result.boxes.cpu().numpy()
#             class_id = boxes.cls[0]
#             conf = boxes.conf[0]
#             xyxy = boxes.xyxy[0]

#             if class_id == 0.0:
          
#               xyxys.append(result.boxes.xyxy.cpu().numpy())
#               confidences.append(result.boxes.conf.cpu().numpy())
#               class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
            
        
#         # Setup detections for visualization
#         detections = sv.Detections(
#                     xyxy=results[0].boxes.xyxy.cpu().numpy(),
#                     confidence=results[0].boxes.conf.cpu().numpy(),
#                     class_id=results[0].boxes.cls.cpu().numpy().astype(int),
#                     )
        
    
#         # Format custom labels
#         self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
#         for _, confidence, class_id, tracker_id
#         in detections]
        
#         # Annotate and display frame
#         frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        
#         return frame
    
    
    
#     def __call__(self):

#         cap = cv2.VideoCapture(self.capture_index)
#         assert cap.isOpened()
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
      
#         while True:
          
#             start_time = time()
            
#             ret, frame = cap.read()
#             assert ret
            
#             results = self.predict(frame)
#             frame = self.plot_bboxes(results, frame)
            
#             end_time = time()
#             fps = 1/np.round(end_time - start_time, 2)
             
#             cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            
#             cv2.imshow('YOLOv8 Detection', frame)
 
#             if cv2.waitKey(5) & 0xFF == 27:
                
#                 break
        
#         cap.release()
#         cv2.destroyAllWindows()
        
        
    
# detector = ObjectDetection(capture_index=1)
# detector()

# import device
# import cv2

# # def select_camera(last_index, camera_name):
# #     for camera_name_index in camera_name:
# #         if camera_name_index == "Logi C270 WebCam":
# #             camera_index= last_index


# def open_camera(index):
#     cap = cv2.VideoCapture(index)
#     return cap

# def main():
#     # print OpenCV version
#     print("OpenCV version: " + cv2.__version__)

#     # Get camera list
#     device_list = device.getDeviceList()
#     index = 0

#     for camera in device_list:
#         print(str(index) + ': ' + camera[0])
        
#         index += 1

#     last_index = index - 1

#     if last_index < 0:
#         print("No device is connected")
#         return

#     # Select a camera
#     # camera_number = select_camera(last_index)
    
#     # Open camera
#     cap = open_camera(0)
#     cap1 = open_camera(1)

#     if cap.isOpened():
#         width = cap.get(3) # Frame Width
#         height = cap.get(4) # Frame Height
#         print('Default width: ' + str(width) + ', height: ' + str(height))

#         while True:
            
#             ret, frame = cap.read()
#             ret1, frame1 = cap1.read()
#             if ret == False or ret1 == False:
#                 print("No camera")
#                 break
#             else:
#                 stitching=cv2.Stitcher.create()
#                 status,frame2=stitching.stitch((frame,frame1))
#                 if status==0:
#                     cv2.imshow('frame',frame2)
#                 else:
#                     print("Error")
#                     break
#             # Display the resulting frame
#             # side_by_side = cv2.hconcat([frame, frame1])
#             # cv2.imshow('frame', side_by_side)

#             # key: 'ESC'
#             key = cv2.waitKey(20)
#             if key == 27:
#                 break

#         cap.release()
#         cap1.release() 
#         cv2.destroyAllWindows() 

# if __name__ == "__main__":
#     main()

import cv2
image1=cv2.imread('image1.jpg')
image2=cv2.imread('image2.jpg')
height1, width1 = image1.shape[:2]
height2, width2 = image2.shape[:2]
print(height1, width1, height2, width2)
stitching=cv2.Stitcher.create()
status,stitched_image=stitching.stitch((image1,image2))
if status==0:
    cv2.imshow('frame',stitched_image)
    cv2.imwrite('stitched_image.jpg',stitched_image)
else:
    print(status)
