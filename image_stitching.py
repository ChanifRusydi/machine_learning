import cv2
import argparse
import time

import platform
import sys

def get_matcher():
    range_width=-1
    matcher_type = 'homography' # 'homography' or 'affine'
    match_conf = 0.3 # only for homography
    if matcher_type == 'affine':
        matcher= cv2.detail_AffineBestOf2NearestMatcher(False, False, match_conf)
    elif range_width== -1:
        matcher= cv2.detail.BestOf2NearestMatcher_create(False, match_conf)

def get_compensator(args):
    compensator = cv.detail.ExposureCompensator_createDefault(cv.detail.ExposureCompensator_GAIN_BLOCKS)
    return compensator

def image_stitching(args, img_names):
    if args.feature_finder == "AKAZE":
        feature_finder = cv2.AKAZE_create()
    elif args.feature_finder == "KAZE":    
        feature_finder=cv2.KAZE_create()
    elif args.feature_finder == "ORB":
        feature_finder=cv2.ORB_create()
    
    

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
        print(type(args.image), args.image)
        img1 = cv2.imread(args.image[0])
        img2 = cv2.imread(args.image[1])
        if args.image is None:
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
        print(args.video)
        video1 = cv2.VideoCapture(args.video[0])
        video2 = cv2.VideoCapture(args.video[1])
        video_status = []
        if not video1.isOpened() or not video2.isOpened():
            print(f'Error: could not open video file "{args.video}"')
            sys.exit()
        # Process video file
        # ...'
        fps1 = video1.get(cv2.CAP_PROP_FPS)
        frame_count1 = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
        duration1 = frame_count1/fps1

        height2, width2 = video2.get(4), video2.get(3)
        fps2 = video2.get(cv2.CAP_PROP_FPS)
        frame_count2 = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
        duration2 = frame_count2/fps2
        print(height2, width2)
        print(f'fps1: {fps1}, frame_count1: {frame_count1}, duration1: {duration1}')
        print(f'fps2: {fps2}, frame_count2: {frame_count2}, duration2: {duration2}')
        frame_count_list = [frame_count1, frame_count2]
        for i in range(min(frame_count_list)):
            ret1, frame1 = video1.read()
            ret2, frame2 = video2.read()
            if not ret1 or not ret2:
                print('Error: could not read frame')
                sys.exit()
            side_by_side = cv2.vconcat([frame1, frame2])
            cv2.imshow('frame', side_by_side)
            if cv2.waitKey(1) == ord('q'):
                break
            # time.sleep(1/fps)


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Stitching')
    parser.add_argument('--mode', default='image', type=str, help='camera or video or image')
    parser.add_argument('--image', type=str,nargs='+',help='path to input image file')
    parser.add_argument('--camera', type=int, default=[0,1], help='camera index')
    parser.add_argument('--video', type=str,nargs='+', help='path to input video file')
    parser.add_argument('--feature_finder', type=str, default='AKAZE', help='AKAZE or KAZE or ORB')
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
