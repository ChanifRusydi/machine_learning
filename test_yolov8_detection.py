import cv2
import os
import argparse
from streamlit_app.yolov8_detect import detect
import streamlit as st
import logging

st.set_page_config(layout="wide")
logging.basicConfig(filename='logfile.txt',filemode='a',format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)
logging = logging.getLogger(__name__)

def check_and_create_folder(path):
    # if os.path.exists(path):
    #     return 'exist'
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False

def check_file_name_index(dir_path):
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.jpg'):
            file_name = file_name.split('.')[0]
            file_name = file_name.split('_')[-1]
            file_indices = int(file_name)
    print(file_indices)
    return file_indices


# parser = argparse.ArgumentParser()
# parser.add_argument('--time', type=str, choices= ['pagi', 'siang', 'sore', 'malam'], default='pagi',help='time of the day')

# args = parser.parse_args()

# if args.time == 'pagi':
#     time_of_day = 'pagi'
# elif args.time == 'siang':
#     time_of_day = 'siang'
# elif args.time == 'sore':
#     time_of_day = 'sore'
# elif args.time == 'malam':
#     time_of_day = 'malam'
time_of_day = st.selectbox('Time of the day', ['pagi', 'siang', 'sore', 'malam'])

logging.info('Time of the day: {}'.format(time_of_day))

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

image_dir_base = './images/'
detect_dir_base = './images/detect/'
with st.container():
    image_placeholder1, image_placeholder2 = st.columns(2)  
    image_placeholder1, image_placeholder2 = st.empty(), st.empty()
    image_placeholder1.header("Image from Video 1")
    image_placeholder2.header("Image 2")
    if video_status1 and video_status2:
        image_placeholder1.selectbox('Kiri atau Kanan', ['Kiri', 'Kanan'], key='selectbox_kiri')
        image_placeholder2.selectbox('Kiri atau Kanan', ['Kiri', 'Kanan'], key='selectbox_kanan')

        image_dir = os.path.join(image_dir_base + time_of_day + '/')
        print(image_dir)
        check_and_create_folder(image_dir)
        open_video1 = cv2.VideoCapture('video1.mp4')
        open_video2 = cv2.VideoCapture('video2.mp4')

        fps1 = open_video1.get(cv2.CAP_PROP_FPS)
        fps2 = open_video2.get(cv2.CAP_PROP_FPS)

        frame_count1 = int(open_video1.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count2 = int(open_video2.get(cv2.CAP_PROP_FRAME_COUNT))

        image_count=min(frame_count1,frame_count2)//min(int(fps1),int(fps2))
        if image_count > 20:
            image_count = 20

        print(image_count)
        logging.info("'FPS 1: {}'.format(fps1), 'FPS 2: {}'.format(fps2)")
        logging.info("'Frame Count 1: {}'.format(frame_count1), 'Frame Count 2: {}'.format(frame_count2)")
        logging.info('Image Count: {}'.format(image_count))
        for i in range(0,image_count):
            open_video1.set(cv2.CAP_PROP_POS_FRAMES, i*fps1)
            open_video2.set(cv2.CAP_PROP_POS_FRAMES, i*fps2)

            status1, frame1 = open_video1.read()
            status2, frame2 = open_video2.read()
            if status1 and status2:
                image_placeholder1.image(frame1, channels='BGR', use_column_width=True)
                image_placeholder2.image(frame2, channels='BGR', use_column_width=True)
                if check_and_create_folder:
                    filename = os.path.join(image_dir + str(i)+'.jpg')
                    cv2.imwrite(filename=filename, img=frame1)
                    if os.listdir(image_dir) == []:
                        filename = os.path.join(image_dir + str(i)+'.jpg')
                        cv2.imwrite(filename=filename, img=frame1)
                    else:
                        last_file_index = check_file_name_index(image_dir)
                        filename = os.path.join(image_dir + str(i+last_file_index)+'.jpg')
                        cv2.imwrite(filename=filename, img=frame2)
            else:
                break
        open_video1.release()
        open_video2.release()

    elif video_status1 and not video_status2:
        open_video1 = cv2.VideoCapture('video1.mp4')
        fps1 = open_video1.get(cv2.CAP_PROP_FPS)
        frame_count1 = int(open_video1.get(cv2.CAP_PROP_FRAME_COUNT))
        image_count = frame_count1//int(fps1)
        logging.info('FPS 1: {}'.format(fps1))
        logging.info('Frame Count 1: {}'.format(frame_count1))
        logging.info('Image Count: {}'.format(image_count))
        for i in range(0,frame_count1,int(fps1)):
            open_video1.set(cv2.CAP_PROP_POS_FRAMES, i)
            status1, frame1 = open_video1.read()
            if status1:
                image_placeholder1.image(frame1, channels='BGR')
            else:
                break
        open_video1.release()
    elif not video_status1 and video_status2:
        image_placeholder1.write('Video 1 Not Found')
        open_video2 = cv2.VideoCapture('video2.mp4')
        fps2 = open_video2.get(cv2.CAP_PROP_FPS)
        frame_count2 = int(open_video2.get(cv2.CAP_PROP_FRAME_COUNT))
        image_count = frame_count2//int(fps2)
        logging.info('FPS 2: {}'.format(fps2))
        logging.info('Frame Count 2: {}'.format(frame_count2))
        logging.info('Image Count: {}'.format(image_count))
        for i in range(0,image_count,int(fps2)):
            open_video2.set(cv2.CAP_PROP_POS_FRAMES, i)
            status2, frame2 = open_video2.read()
            if status2:
                image_placeholder2.image(frame2, channels='BGR')
            else:
                break
        open_video2.release()
    else:
        image_placeholder1.write('No Video 1')
        image_placeholder2.write('No Video 2')

with st.container():
    time_of_day_select = st.selectbox('Pilih Waktu', ['pagi', 'siang', 'sore', 'malam'])
    dir_name = os.path.join(image_dir_base + time_of_day_select)
    if not os.path.exists(dir_name):
        st.write('Folder tidak ditemukan')

    else:
        button = st.button('Detect')
        st.header('Hasil Deteksi')
        if button:
            detect_dir_status = check_and_create_folder(detect_dir_base + time_of_day_select)
            if detect_dir_status:
                st.write('Folder detect berhasil dibuat')
            file_list = sorted([f for f in os.listdir(dir_name) if f.endswith('.jpg')])
            for i in file_list:
                image_directory = os.path.join(dir_name, i)
                print(image_directory)
                image = cv2.imread(os.path.join(dir_name, i))
                st.image(image, channels='BGR', use_column_width=True)
                image = detect(image)
                st.image(image, channels='BGR', use_column_width=True)
                filename = os.path.join(detect_dir_base + time_of_day_select + i + '.jpg')
                cv2.imwrite(filename=filename, img=image)

    

# if 'counter' not in st.session_state:
#     st.session_state.counter = 0


# if not os.path.exists(detect_dir):
#     os.makedirs(detect_dir)

# cols = st.columns(2)
# image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])


# for index, image_file in enumerate(image_files):
#     image_name = os.path.join(image_dir, image_file)
#     print('Image Name', image_name, index)
#     image = cv2.imread(image_name)
#     show_image = st.empty()
#     show_image.image(image, channels='BGR')
    
    
#     cv2.imshow(image_name, image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     status, image = detect(image)
#     if status == 1:
#         cv2.imwrite('images_detected/'+image_name, image)
#     else:
#         cv2.imwrite('images_not_detected/'+image_name, image)

