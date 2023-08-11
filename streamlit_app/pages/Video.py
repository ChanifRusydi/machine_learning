import streamlit as st
import cv2

import logging
from yolov8_detect import detect
from image_stitching import image_stitching

logging.basicConfig(filename='logfile.txt',filemode='a',format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)
logging.info('Enter Video Page')

st.title("Video Page")

video_status1 = False
video_status2 = False

with st.container():
    video1_placeholder, video2_placeholder = st.columns(2)
    video_upload1 = video1_placeholder.file_uploader("Upload Video 1", type=['mp4'])
    video_upload2 = video2_placeholder.file_uploader("Upload Video 2", type=['mp4'])
    video1_placeholder.header("Video 1")
    video2_placeholder.header("Video 2")
    if video_upload1 is not None :
            with open('video1.mp4', 'wb') as f:
                f.write(video_upload1.getbuffer())
             
            video_status1 = True
    else:
        logging.info('Uploaded Video Not Found')
        # open_video1 = cv2.VideoCapture('../../image1_60_left.mp4')
    
    if video_upload2 is not None:
        with open('video2.mp4', 'wb') as f:
            f.write(video_upload2.getbuffer())
        video_status2 = True   
    video1_placeholder.video(video_upload1)
    video2_placeholder.video(video_upload2)

    # open_video1 = cv2.VideoCapture('../../image1_60_left.mp4')   
    # open_video2 = cv2.VideoCapture('../../image1_60_right.mp4')
    
    # ret1, frame1 = open_video1.read()
    # ret2, frame2 = open_video2.read()
            
            
    #         video1_placeholder.image(frame1, channels="BGR")
    #         video2_placeholder.image(frame2, channels="BGR")
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    # elif open_video1.open() and not open_video2.open():
    #     while open_video1.isOpened():
    #         ret, frame1 = open_video1.read()
    #         if not ret:
    #             break
    #         video1_placeholder.image(frame1, channels="BGR")
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    # elif not open_video1.open() and open_video2.open():
    #     while open_video2.isOpened():
    #         ret, frame2 = open_video2.read()
    #         if not ret:
    #             break
    #         video2_placeholder.image(frame2, channels="BGR")
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    # else:
    #     st.write("No Video")                   
    # open_video1.release()
    # open_video2.release()

with st.container():
    side_by_side_placeholder = st.empty()
    side_by_side_placeholder.header("Side by Side Video")
    if video_status1 and video_status2:
        open_video1 = cv2.VideoCapture('video1.mp4')
        open_video2 = cv2.VideoCapture('video2.mp4')
        while True:
            ret1, frame1 = open_video1.read()
            ret2, frame2 = open_video2.read()
            if not ret1 or not ret2:
                break
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)
            if frame1.shape != frame2.shape:
                side_by_side_placeholder.subheader("Please open video with same shape")
            else:
                image_stitching_status, image = image_stitching(frame1, frame2)
                if image_stitching_status == -1 and image is None:
                    image = cv2.hconcat([frame1, frame2])
                status,image = detect(image)
                side_by_side_placeholder.image(image, channels="RGB")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        open_video1.release()
        open_video2.release()
    else:
        side_by_side_placeholder.subheader("Please upload both video")