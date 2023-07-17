import cv2
import argparse
import time

import platform
import sys

def image_stitching(args):
    cv2.AKAZE_create()
    cv2.KAZE_create()
    cv2.ORB_create()

def open_camera(index):
    cap = cv2.VideoCapture(index)
    return cap

def main(args):
    # print OpenCV version
    print("OpenCV version: " + cv2.__version__)

    if args.mode == 'image':
        # Read input image
        if args.image is None:
            print('Error: --image argument is required when --mode is set to "image"')
            return
        img = cv2.imread(args.image)
        if img is None:
            print(f'Error: could not read image file "{args.image}"')
            return
        # Process input image
        # ...
    elif args.mode == 'camera':
        # Open camera
        cap = open_camera(0)
        cap1 = open_camera(1)

        if cap.isOpened():
            width = cap.get(3) # Frame Width
            height = cap.get(4) # Frame Height
            # Process camera input
            # ...
    elif args.mode == 'video':
        # Open video file
        if args.video is None:
            print('Error: --video argument is required when --mode is set to "video"')
            return 'Error: --video argument is required when --mode is set to "video"'
        
        video1 = cv2.VideoCapture(args.video[0])
        if not cap.isOpened():
            print(f'Error: could not open video file "{args.video}"')
            return
        # Process video file
        # ...'
        fps = cap.get(cv2.CAP_PROP_FPS)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Stitching')
    parser.add_argument('--mode', default='image', type=str, help='camera or video or image')
    parser.add_argument('--image', type=str,nargs='+',help='path to input image file')
    parser.add_argument('--camera', type=int, default=[0,1], help='camera index')
    parser.add_argument('--video', type=str, help='path to input video file')
    args = parser.parse_args()
    print(args.mode)
    print(args.image)
    print(args.camera)
    main(args)
# import cv2
# image1=cv2.imread('image1.jpg')
# image2=cv2.imread('image2.jpg')
# height1, width1 = image1.shape[:2]
# height2, width2 = image2.shape[:2]
# print(height1, width1, height2, width2)
# stitching=cv2.Stitcher.create()
# status,stitched_image=stitching.stitch((image1,image2))
# if status==0:
#     cv2.imshow('frame',stitched_image)
#     cv2.imwrite('stitched_image.jpg',stitched_image)
# else:
#     print(status)
