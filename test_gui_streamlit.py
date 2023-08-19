import streamlit as st
import cv2
import logging
st.set_page_config(layout="wide")
logging.basicConfig(filename='logfile.txt',filemode='a',format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)
logging.info('Start Python Streamlit App')
logger = logging.getLogger(__name__)
 

def show_result_image(image):
    st.image(image, channels="BGR")


def image_stitching(image1, image2):
    return status, image

def single_camera_mode():
    st.write("Single Camera Mode")

def double_camera_mode():
    st.write("Double Camera Mode")

def main():
    # st.set_page_config(layout="wide")
    # column1, column2 = st.columns(2)
   
    # st.title("OpenCV and Streamlit")
    
    # st.caption("OpenCV and Streamlit")
    # cap1 = cv2.VideoCapture(0)
    # cap2 = cv2.VideoCapture(1)
    # frame_placeholder1 = column1.empty()
    # frame_placeholder2 = column2.empty()
    # # frame_placeholder1 = st.empty()
    # # frame_placeholder2 = st.empty()
    # stop_button = st.button("Stop")
    # while cap1.isOpened() and not stop_button:
    #     ret, frame1 = cap1.read()
    #     ret, frame2 = cap2.read()
    #     if not ret:
    #         break
    #     frame_placeholder1.image(frame1, channels="BGR")
    #     frame_placeholder2.image(frame2, channels="BGR")

    #     status, result = image_stitching(frame1, frame2)
    #     # if status= -1:
    #     #     result = cv.imread('result.jpg')
    #     # frame_placeholder3.image(result, channels="BGR")
    #     if cv2.waitKey(1) & 0xFF == ord('q') or stop_button:
    #         break
    # cap1.release()
    # cap2.release()
    # cv2.destroyAllWindows()
    # st.set_page_config(page_title="Streamlit WebCam App")
    # st.title("Webcam Display Steamlit App")
    # st.caption("Powered by OpenCV, Streamlit")
    # cap = cv2.VideoCapture(0)
    # frame_placeholder = st.empty()
    # stop_button_pressed = st.button("Stop")
    # while cap.isOpened() and not stop_button_pressed:
    #     ret, frame = cap.read()
    #     if not ret:
    #         st.write("Video Capture Ended")
    #         break
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     frame_placeholder.image(frame,channels="RGB")
    #     if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    # camera1 = cv2.VideoCapture(0)
    # camera2 = cv2.VideoCapture(1)
    # camera1_placeholder, camera2_placeholder = st.columns([0.5, 0.5])
    # counter = 0
    # while True:
    #     status1 = camera1.grab()
    #     status2 = camera2.grab()
        # print(status1,status2)
        # counter+=1

        # break
        # if not status1:
        #     print("frame1 is None")
        # elif status1 or status2:
        #     print("frame2 is None")
        #     _,frame1 = camera1.retrieve()
        #     print(type(frame1), frame1)
            # frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            # camera1_placeholder.image(frame1, channels="BGR")
        # else:
        #     print("frame1 and frame2 is not None"
        #     frame1 = cv2.cvtColor(camera1.retrieve(), cv2.COLOR_BGR2RGB)
        #     camera1_placeholder.image(frame1, channels="BGR")
        #     frame2 = cv2.cvtColor(camera2.retrieve(), cv2.COLOR_BGR2RGB)
        #     camera2_placeholder.image(frame2, channels="BGR")
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break
        camera_mode = st.selectbox("Select Camera Mode", ["Single Camera Mode", "Double Camera Mode"])
        if camera_mode == "Single Camera Mode":
            single_camera_mode()
        elif camera_mode == "Double Camera Mode":
            double_camera_mode()

if __name__ == "__main__":
    main()