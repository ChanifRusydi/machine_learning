import cv2
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-dual_camera", default=False, help="Use dual camera")


 # if dual camera is True, then we are stitching the images from two cameras
if args["dual_camera"]:
    # capture the images from the cameras
    camera1 = cv2.VideoCapture(0)
    camera2 = cv2.VideoCapture(1)
    # grab the frames from the cameraas
    (grabbed1, frame1) = camera1.read()
    (grabbed2, frame2) = camera2.read()
    # stitch the frames together to form the panorama
    stitch= Stitcher()
    (result, vis) = stitch.stitch([frame1, frame2], showMatches=True)
    # show the images
    cv2.imshow("Keypoint Matches", vis)
    cv2.imshow("Result", result)
    # run object detection on the image
    
    cv2.waitKey(0)
else:
    # run object detection on the image
    image = cv2.imread("images/3.jpg")
    image = imutils.resize(image, width=400)
    
