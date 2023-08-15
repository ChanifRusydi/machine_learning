# import cv2
# import matplotlib.pyplot as plt
# import time
# image1= cv2.imread("image1_60_left.jpg", flags=cv2.IMREAD_GRAYSCALE)
# image2= cv2.imread("image1_60_right.jpg",flags=cv2.IMREAD_GRAYSCALE)
# print(image1.shape)
# print(image2.shape)

# # Initiate SIFT detector
# sift = cv2.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(image1,None)
# kp2, des2 = sift.detectAndCompute(image2,None)
# # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50) # or pass empty dictionary
# flann = cv2.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des1,des2,k=2)
# # Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         matchesMask[i]=[1,0]
#         draw_params = dict(matchColor = (0,255,0),
# singlePointColor = (255,0,0),
# matchesMask = matchesMask,
# flags = cv2.DrawMatchesFlags_DEFAULT)
# img3 = cv2.drawMatchesKnn(image1,kp1,image2,kp2,matches,None,**draw_params)
# plt.imshow(img3,),plt.show()



# BRISK = cv2.BRISK_create()
# time_start=time.time()
# keypoints1, descriptors1 = BRISK.detectAndCompute(image1, None)
# keypoints2, descriptors2 = BRISK.detectAndCompute(image2, None)

# # create BFMatcher object
# # BFMatcher = cv2.BFMatcher(normType = cv2.NORM_HAMMING,
# #                          crossCheck = True)

# # # Matching descriptor vectors using Brute Force Matcher
# # matches = BFMatcher.match(queryDescriptors = descriptors1,
# #                           trainDescriptors = descriptors2)
# time_end=time.time()
# # Sort them in the order of their distance
# matches = sorted(matches, key = lambda x: x.distance)

# # Draw first 15 matches
# output = cv2.drawMatches(img1 = image1,
#                         keypoints1 = keypoints1,
#                         img2 = image2,
#                         keypoints2 = keypoints2,
#                         matches1to2 = matches[:100],
#                         outImg = None,
#                         flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# plt.imshow(output)
# plt.show()
# print(time_end-time_start)

# import cv2
# import time
# import os
# # Importing all necessary libraries
# # import cv2
# # import os

# # Read the video from specified path
# cam = cv2.VideoCapture(r"C:\Users\User\Documents\machine_learning\Video1.mp4")

# # frame
# currentframe = 0

# fps = cam.get(cv2.CAP_PROP_FPS)
# frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
# print(fps, frame_count)
# for i in range(frame_count):
#     # reading from frame
#     ret,frame = cam.read()
#     if ret :
# 		# if video is still left continue creating images
#         name = os.path.join(r"C:\Users\User\Documents\machine_learning\yolov7\Video", str(currentframe) + '.jpg') 
#         print ('Creating...' + name)
#         cv2.imwrite(name, frame)
# 		# writing the extracted images
      
#         cv2.imshow(f"frame ",frame)
#         if cv2.waitKey(0) & 0xFF == ord('q'):
#             break
#         # cv2.imshow("frame",frame)
#         # time.sleep(0.1)
# 		# increasing counter so that it will
# 		# show how many frames are created
#         currentframe += 1
#     else:
#         break

# # Release all space and windows once done
# cam.release()
# cv2.destroyAllWindows()


# def video_to_frames(input_loc, output_loc):
#     """Function to extract frames from input video file
#     and save them as separate frames in an output directory.
#     Args:
#         input_loc: Input video file.
#         output_loc: Output directory to save the frames.
#     Returns:
#         None
#     """
    
#     # Log the time
#     time_start = time.time()
#     # Start capturing the feed
#     cap = cv2.VideoCapture(input_loc)
#     # Find the number of frames
#     video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
#     print ("Number of frames: ", video_length)
#     count = 0
#     print ("Converting video..\n")
#     # Start converting the video
#     while cap.isOpened():
#         # Extract the frame
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         # Write the results back to output location.
#         cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
#         count = count + 1
#         # If there are no more frames left
#         if (count > (video_length-1)):
#             # Log the time again
#             time_end = time.time()
#             # Release the feed
#             cap.release()
#             # Print stats
#             print ("Done extracting frames.\n%d frames extracted" % count)
#             print ("It took %d seconds forconversion." % (time_end-time_start))
#             break

# if __name__=="__main__":

#     input_loc = 'Video1.mp4'
#     output_loc = r'C:\Users\User\Documents\machine_learning\yolov7\Video'
#     video_to_frames(input_loc, output_loc)

# Initiate SIFT detector
# from __future__ import print_function
# import cv2 as cv
# import numpy as np
# import argparse

# parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
# parser.add_argument('--input1', help='Path to input image 1.', default='image1_60_left.jpg')
# parser.add_argument('--input2', help='Path to input image 2.', default='image1_60_right.jpg')
# args = parser.parse_args()

# img1 = cv.imread(cv.samples.findFile(args.input1), cv.IMREAD_GRAYSCALE)
# img2 = cv.imread(cv.samples.findFile(args.input2), cv.IMREAD_GRAYSCALE)
# if img1 is None or img2 is None:
#     print('Could not open or find the images!')
#     exit(0)

# #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
# minHessian = 400
# detector = cv.ORB_create()
# keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
# keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

# #-- Step 2: Matching descriptor vectors with a FLANN based matcher
# # Since SURF is a floating-point descriptor NORM_L2 is used
# matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
# knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

# #-- Filter matches using the Lowe's ratio test
# ratio_thresh = 0.7
# good_matches = []
# for m,n in knn_matches:
#     if m.distance < ratio_thresh * n.distance:
#         good_matches.append(m)

# #-- Draw matches
# img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
# cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# #-- Show detected matches
# cv.imshow('Good Matches', img_matches)

# cv.waitKey()

# import os
# import sys
# import random
# import argparse
# import streamlit as st
# import logging

# logging.basicConfig(filename='logfile.txt',filemode='a',format='%(asctime)s - %(message)s',
#                     datefmt='%d-%b-%y %H:%M:%S',
#                     level=logging.INFO)
# logging = logging.getLogger(__name__)

# parser = argparse.ArgumentParser(description='This app lists animals')

# parser.add_argument('--animal', action='append', default=[],
#                     help="Add one or more animals of your choice")
# sort_order_choices = ('up', 'down', 'random')
# parser.add_argument('--sort', choices=sort_order_choices, default='up',
#                     help='Animal sort order (default: %(default)s)')
# parser.add_argument('--uppercase', action='store_true',
#                     help='Make the animals bigger!')
# try:
#     args = parser.parse_args()
# except SystemExit as e:
#     # This exception will be raised if --help or invalid command line arguments
#     # are used. Currently streamlit prevents the program from exiting normally
#     # so we have to do a hard exit.
#     os._exit(e.code)

# st.title("Command line example app")
# st.markdown("""
# Your current command line is:
# ```
# {}
# ```
# A double dash (`--`) is used to separate streamlit arguments from app arguments.
# As a result
# ```
# streamlit run command_line.py --help
# ```
# will show the help for streamlit and
# ```
# streamlit run command_line.py -- --help
# ```
# will show the help for this app. Try
# ```
# streamlit run command_line.py -- --animal dog --animal cat --sort down
# ```
# to see it in action.
# """.format(sys.argv))

# # Built in animals
# animals = ['Albatross', 'Bison', 'Dragonfly', 'Shark', 'Zebra']

# # Add one or more animals supplied on command line
# animals += args.animal
# logging.info('Animals: %s', animals)


# # Set default sort order from command line option
# sort_order = st.selectbox("Sort order", sort_order_choices,
#                           sort_order_choices.index(args.sort))

# logging.info('Sort order: %s', sort_order)
# if sort_order == 'up':
#     animals.sort()
# elif sort_order == 'down':
#     animals.sort(reverse=True)
# elif sort_order == 'random':
#     random.shuffle(animals)
# else:
#     # This can't happen unless you add more values to sort_order_choices
#     raise ValueError("Invalid sort order")

# # Set checkbox default from command line
# uppercase_animals = st.checkbox("Uppercase", args.uppercase)
# logging.info('Uppercase: %s', uppercase_animals)
# if uppercase_animals:
#     animals = [animal.upper() for animal in animals]

# # Show the results
# st.header("You list of animals")
# st.dataframe(animals)

import cv2
import os

base_dir = './images/'

video = cv2.VideoCapture('Video1.mp4')
fps = video.get(cv2.CAP_PROP_FPS)
print(fps)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(frame_count)
durattion = frame_count/fps
print(durattion)
image_count = int(frame_count / fps)
print(image_count)
# for i in range(0,frame_count,int(fps)):
#     video.set(1,i)
#     ret, frame = video.read()
#     cv2.imshow('frame',frame)
#     cv2.waitKey(0)
#     last_index = 20
#     filename = os.path.join(base_dir + str(i+last_index) + '.jpg')
#     print(filename)
    # cv2.imwrite(filename=filename, img=frame)
import time
dir_name = './images/pagi'
def extract_integer(filename):
    return int(filename.split('.')[0].split('_')[1])
integer = extract_integer('19.jpg')
print(integer)
# file_list = sorted(os.listdir(dir_name), key=extract_integer)
# for i in file_list:
#     time.sleep(0.5)
#     print(i)

# video.release()