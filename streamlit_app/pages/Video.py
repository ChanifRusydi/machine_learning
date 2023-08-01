import streamlit as st
import cv2

st.title("Video Page")

with st.container():
    video1_placeholder, video2_placeholder = st.columns(2)
    image_upload1 = video1_placeholder.file_uploader("Upload Video 1", type=['mp4'])
    image_upload2 = video2_placeholder.file_uploader("Upload Video 2", type=['mp4'])
    video1_placeholder.header("Video 1")
    video2_placeholder.header("Video 2")
    open_video1 = cv2.VideoCapture('../../image1_60_left.mp4')   
    open_video2 = cv2.VideoCapture('../../image1_60_right.mp4')
    if open_video1.open() and open_video2.open():
        while cap1.isOpened() and cap2.isOpened():
            ret, frame1 = cap1.read()
            ret, frame2 = cap2.read()
            if not ret:
                break
            video1_placeholder.image(frame1, channels="BGR")
            video2_placeholder.image(frame2, channels="BGR")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    elif open_video1.open() and not open_video2.open():
        while cap1.isOpened():
            ret, frame1 = cap1.read()
            if not ret:
                break
            video1_placeholder.image(frame1, channels="BGR")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    elif not open_video1.open() and open_video2.open():
        while cap2.isOpened():
            ret, frame2 = cap2.read()
            if not ret:
                break
            video2_placeholder.image(frame2, channels="BGR")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        st.write("No Video")                   
    cap1.release()
    cap2.release()

