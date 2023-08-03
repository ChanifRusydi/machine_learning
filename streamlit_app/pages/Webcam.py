import streamlit as st
import cv2
import numpy as np
from streamlit_extras.switch_page_button import switch_page

st.title("Webcam Page")
back_button  = st.button("Back to Home Page")
if back_button:
    switch_page("streamlit_app")
with st.container():
    camera1_placeholder, camera2_placeholder = st.columns(2)
    camera1_placeholder.header("Camera 1")
    camera2_placeholder.header("Camera 2")

    camera1 = cv2.VideoCapture(0)
    camera2 = cv2.VideoCapture(1)
    
    camera1_placeholder, camera2_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")
    
    ret1, frame1 = camera1.read()
    ret2, frame2 = camera2.read()
    while ret1 and ret2 and not stop_button_pressed:
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame,channels="RGB")

        if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
            camera1.release()
            break
    while ret1 and not stop_button_pressed:
        ret1, frame1 = camera1.read()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        camera1_placeholder.image(frame1, channels="BGR")
        if cv2.waitKey(1) & 0xFF == ord('q') or stop_button_pressed:
            camera1.release()
            break
    while ret2 and not stop_button_pressed:
        ret2, frame2 = camera2.read()
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        camera2_placeholder.image(frame2, channels="BGR")
        if cv2.waitKey(1) & 0xFF == ord('q') or stop_button_pressed:
            break

camera1.release()
cv2.destroyAllWindows()
