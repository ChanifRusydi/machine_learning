import streamlit as st
import cv2
from tempfile import NamedTemporaryFile
import logging

logging.basicConfig(filename='logfile.txt',filemode='a',format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)
logging.info('Enter Video Page')

st.title("Video Page")

with st.container():
    video1_placeholder, video2_placeholder = st.columns(2)
    video_upload1 = video1_placeholder.file_uploader("Upload Video 1", type=['mp4'])
    video_upload2 = video2_placeholder.file_uploader("Upload Video 2", type=['mp4'])
    video1_placeholder.header("Video 1")
    video2_placeholder.header("Video 2")
    if video_upload1 is not None :
        with NamedTemporaryFile(suffix='mp4') as tmp1:
            tmp1.write(video_upload1.read())
            open_video1 = cv2.VideoCapture(tmp1.name)
    else:
        logging.info('Uploaded Video Not Found')
        open_video1 = cv2.VideoCapture('../../image1_60_left.mp4')
    
    if video_upload2 is not None:
        with NamedTemporaryFile(suffix='mp4') as tmp2:
            tmp2.write(video_upload2.read())
            open_video2 = cv2.VideoCapture(tmp2.name)
    
    video1_placeholder.video(video_upload1)
    video2_placeholder.video(video_upload2)

    # open_video1 = cv2.VideoCapture('../../image1_60_left.mp4')   
    # open_video2 = cv2.VideoCapture('../../image1_60_right.mp4')
    ret1, frame1 = open_video1.read()
    ret2, frame2 = open_video2.read()
            
            
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

