import cv2
import os
import argparse
from streamlit_app.yolov8_detect import detect
import streamlit as st
import logging

logging.basicConfig(filename='logfile.txt',filemode='a',format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)
logging = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--time', type=str, choices= ['pagi', 'siang', 'sore', 'malam'], default='pagi',help='time of the day')

args = parser.parse_args()

if args.time == 'pagi':
    time_of_day = 'pagi'
elif args.time == 'siang':
    time_of_day = 'siang'
elif args.time == 'sore':
    time_of_day = 'sore'
elif args.time == 'malam':
    time_of_day = 'malam'


video_status1 = False
video_status2 = False

with st.container():
    video1_placeholder, video2_placeholder = st.columns(2)
    video_upload1 = video1_placeholder.file_uploader("Upload Video 1", type=['mp4'])
    video_upload2 = video2_placeholder.file_uploader("Upload Video 2", type=['mp4'])
    video1_placeholder.header("Video 1")
    video2_placeholder.header("Video 2")
    if video_upload1 is not None :
            with open('video1.mp4', 'wb') as f:
                f.write(video_upload1.getbuffer())
             
            video_status1 = True
    else:
        logging.info('Uploaded Video Not Found')
        # open_video1 = cv2.VideoCapture('../../image1_60_left.mp4')
    
    if video_upload2 is not None:
        with open('video2.mp4', 'wb') as f:
            f.write(video_upload2.getbuffer())
        video_status2 = True   
    video1_placeholder.video(video_upload1)
    video2_placeholder.video(video_upload2)

with st.container():
    image_placeholder1, image_placeholder2 = st.columns(2)
    if video_status1 and video_status2:


image_dir = './images'
detect_dir = './images/detect'

if not os.path.exists(detect_dir):
    os.makedirs(detect_dir)


image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
for image_file in image_files:
    image_name = os.path.join(image_dir, image_file)
    image = cv2.imread(image_name)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # status, image = detect(image)
    # if status == 1:
    #     cv2.imwrite('images_detected/'+image_name, image)
    # else:
    #     cv2.imwrite('images_not_detected/'+image_name, image)

