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

camera1 = None
camera2 = None

st.title("Webcam Page")
back_button  = st.button("Back to Home Page")
if back_button:
    switch_page("streamlit_app")
camera1 = cv2.VideoCapture(0)
camera2 = cv2.VideoCapture(1)
ret1, frame1 = camera1.read()
ret2, frame2 = camera2.read()
# stop_button = st.button("Stop")

# while ret1 and ret2:
#     frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
#     frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

# while ret1 and not stop_button_pressed:
#     ret1, frame1 = camera1.read()
#     frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
#     camera1_placeholder.image(frame1, channels="BGR")
#     if cv2.waitKey(1) & 0xFF == ord('q') or stop_button_pressed:
#         camera1.release()
#         break
# while ret2 and not stop_button_pressed:
#     ret2, frame2 = camera2.read()
#     frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
#     camera2_placeholder.image(frame2, channels="BGR")
#     if cv2.waitKey(1) & 0xFF == ord('q') or stop_button_pressed:
#         break
    
with st.container():
    camera1_placeholder, camera2_placeholder = st.columns(2)
    camera1_placeholder, camera2_placeholder = st.empty(), st.empty()
    camera1_placeholder.header("Camera 1")
    camera2_placeholder.header("Camera 2")

    
    while ret1 and ret2 and not stop_button:
        camera1_placeholder.image(frame1)
        camera2_placeholder.image(frame2)

        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button:
            camera1.release()
            camera2.release()
            break
    while ret1 and not stop_button:
        camera1_placeholder.image(frame1)
    while ret2 and not stop_button:
        camera2_placeholder.image(frame2)
    


    
with st.container():
    camera_side_by_side_placeholder = st.empty()
    if camera1 is None:
        print("camera1 is None")
        logging.info('Cant open camera1')
        logging
    if camera2 is None:
        print("camera2 is None")
        logging.info('Cant open camera2')

    if camera1 is None or camera2 is None:
        st.write("Cant Sitch Camera")
    else:

        image = cv2.hconcat([frame1, frame2])
        # status, image_detect = detect(image)
        camera_side_by_side_placeholder.image(image, channels="BGR")

camera1.release()
cv2.destroyAllWindows()
