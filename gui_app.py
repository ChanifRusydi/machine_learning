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
    camera1 = cv2.VideoCapture(0)
    camera2 = cv2.VideoCapture(1)
    with st.container():
        camera1_placeholder, camera2_placeholder = st.columns([0.5, 0.5])
        
        camera1_placeholder.header("Camera 1")
        camera2_placeholder.header("Camera 2")
        # ret1, frame1 = camera1.read()
        # ret2, frame2 = camera2.read()
        while True:
            # if status1 and status2:
            #     _, frame1 = camera1.retrieve()
            #     _, frame2 = camera2.retrieve()
            #     break
            # elif status1 and not status2:
            #     _, frame1 = camera1.retrieve()
            #     camera1_placeholder.image(frame1, use_column_width=True)
            #     if cv2.waitKey(1) & 0xFF == ord("q") or stop_button:
            #         camera1.release()
            #         break
            #     continue
            # elif status2 and not status1:
            #     _, frame2 = camera2.retrieve()
            #     camera2_placeholder.image(frame2)
            #     if cv2.waitKey(1) & 0xFF == ord("q") or stop_button:
            #         camera2.release()
            #         break
            camera1_placeholder, camera2_placeholder = st.empty(), st.empty()
            ret1, frame1 = camera1.read()
            ret2, frame2 = camera2.read()
            if ret1 and ret2:
                camera1_placeholder.image(frame1, use_column_width=True)
                camera2_placeholder.image(frame2, use_column_width=True)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    camera1.release()
                    break
            else:
                camera1_placeholder.write("Camera 1 is not available")
                camera2_placeholder.write("Camera 2 is not available")

if __name__ == "__main__":
    main()