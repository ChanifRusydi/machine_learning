import streamlit as st
import cv2
import numpy as np
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.stateful_button import button as stateful_button
from yolov8_detect import detect
import logging

logging.basicConfig(filename='logfile.txt',filemode='a',format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger(__name__)

frame1 = None
frame2 = None

st.title("Webcam Page")
back_button  = st.button("Back to Home Page")
if back_button:
    switch_page("streamlit_app")
camera1 = cv2.VideoCapture(0)
camera2 = cv2.VideoCapture(1)

stop_button = st.button("Stop") 


    
with st.container():
    camera1_placeholder, camera2_placeholder = st.columns([0.5, 0.5])
    camera1_placeholder, camera2_placeholder = st.empty(), st.empty()
    camera1_placeholder.header("Camera 1")
    camera2_placeholder.header("Camera 2")
    ret1, frame1 = camera1.read()
    ret2, frame2 = camera2.read()
    while camera1.isOpened() and camera2.isOpened():
        camera1_placeholder.image(frame1)
        camera2_placeholder.image(frame2)
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button:
            camera1.release()
            camera2.release()
            break
    while camera1.isOpened() and not camera2.isOpened():
        camera1_placeholder.image(frame1)
        camera2_placeholder.write("Camera 2 is not available")
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button:
            camera1.release()
            break
    while camera2.isOpened() and not camera1.isOpened():
        camera2_placeholder.image(frame2)
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button:
            camera2.release()
            break
    while not camera1.isOpened() and not camera2.isOpened():
        camera1_placeholder.write("Camera 1 is not available")
        camera2_placeholder.write("Camera 2 is not available")
        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button:
            break

with st.container():
    camera_side_by_side_placeholder = st.empty()
    if frame1 is None:
        print("camera1 is None")
        logging.info('Cant open camera1')
        logging
    if frame2 is None:
        print("camera2 is None")
        logging.info('Cant open camera2')

    if camera1 is None or camera2 is None:
        st.write("Cant Sitch Camera")
    else:
        while camera1.isOpened() and camera2.isOpened():
            status1, frame1 = camera1.read()
            status2, frame2 = camera2.read()
        image = cv2.hconcat([frame1, frame2])
        # status, image_detect = detect(image)
        camera_side_by_side_placeholder.image(image, channels="BGR")


