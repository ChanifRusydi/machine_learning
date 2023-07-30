import streamlit as st
import cv2
import logging

logging.basicConfig(filename='logfile.txt',filemode='a',format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)
logging.info('Start Python Streamlit App')
logger = logging.getLogger(__name__)
 

def show_result_image(image):
    st.image(image, channels="BGR")


def image_stitching(image1, image2):
    return status, image

def main():
    st.set_page_config(layout="wide")
    column1, column2 = st.columns(2)
   
    st.title("OpenCV and Streamlit")
    
    st.caption("OpenCV and Streamlit")
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    frame_placeholder1 = column1.empty()
    frame_placeholder2 = column2.empty()
    # frame_placeholder1 = st.empty()
    # frame_placeholder2 = st.empty()
    stop_button = st.button("Stop")
    while cap1.isOpened() and not stop_button:
        ret, frame1 = cap1.read()
        ret, frame2 = cap2.read()
        if not ret:
            break
        frame_placeholder1.image(frame1, channels="BGR")
        frame_placeholder2.image(frame2, channels="BGR")

        status, result = image_stitching(frame1, frame2)
        if status= -1:
            result = cv.imread('result.jpg')
        frame_placeholder3.image(result, channels="BGR")
        if cv2.waitKey(1) & 0xFF == ord('q') or stop_button:
            break
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()