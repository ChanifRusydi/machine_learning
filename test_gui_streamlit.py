import streamlit as st
import cv2

def main():
    st.set_page_config(layout="wide")
    st.title("OpenCV and Streamlit")
    
    st.caption("OpenCV and Streamlit")
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    stop_button = st.button("Stop")
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break
        frame_placeholder.image(frame, channels="BGR")
        if cv2.waitKey(1) & 0xFF == ord('q') or stop_button:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()