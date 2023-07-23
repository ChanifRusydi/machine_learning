import streamlit as st
import cv2

def main():
    column1, column2 = st.columns()
    st.set_page_config(layout="wide")
    st.title("OpenCV and Streamlit")
    
    st.caption("OpenCV and Streamlit")
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    frame_placeholder1 = column1.empty()
    frame_placeholder2 = column2.empty()
    # frame_placeholder1 = st.empty()
    # frame_placeholder2 = st.empty()
    stop_button = st.button("Stop")
    while cap.isOpened() and not stop_button:
        ret, frame1 = cap1.read()
        ret, frame2 = cap2.read()
        if not ret:
            break
        frame_placeholder1.image(frame1, channels="BGR")
        frame_placeholder2.image(frame2, channels="BGR")
        if cv2.waitKey(1) & 0xFF == ord('q') or stop_button:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()