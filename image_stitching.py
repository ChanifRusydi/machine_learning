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

 