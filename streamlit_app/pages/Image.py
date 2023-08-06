import streamlit as st
import cv2
import numpy as np
from yolov8_detect import detect
from image_stitching import image_stitching


image1 = None
image2 = None
st.title("Image Page")
with st.container():
    image1_placeholder, image2_placeholder = st.columns(2)
    image_upload1 = image1_placeholder.file_uploader("Upload Image 1", type=['jpg', 'png'])
    image_upload2 = image2_placeholder.file_uploader("Upload Image 2", type=['jpg', 'png']) 

    if image_upload1 is not None:
        image1 = cv2.imdecode(np.fromstring(image_upload1.read(), np.uint8), 1)
    else:
        image1 = cv2.imread('../../image1_60_left.jpg')
    if image_upload2 is not None:
        image2 = cv2.imdecode(np.fromstring(image_upload2.read(), np.uint8), 1)
    else:
        image2 = cv2.imread('../../image1_60_right.jpg')

    image1_placeholder.header("Image 1")
    image2_placeholder.header("Image 2")
    if image1 is not None:
        with image1_placeholder:
            print(type(image1))
            st.image(image1, channels="BGR")
    if image2 is not None:
        with image2_placeholder:
            st.image(image2, channels="BGR")
st.header("Side by Side Image")

with st.container():
    image_side_by_side_placeholder = st.empty()
    if image1 is None:
        print("image1 is None")
    if image2 is None:
        print("image2 is None")

    # image = cv2.hconcat([image1, image2])
    # image_side_by_side_placeholder.image(image, channels="BGR")
    if image1 is None or image2 is None:
        image_side_by_side_placeholder.subheader("Please upload both images")
    else:
        if image1.shape != image2.shape:
            image_side_by_side_placeholder.subheader("Please upload images with same shape")
        else:
            image_stitching_status, image = image_stitching(image1, image2)
            if image_stitching_status == -1 and image is None:
                image = cv2.hconcat([image1, image2])
            status, image_detect = detect(image)
            image_side_by_side_placeholder.image(image_detect, channels="BGR")
    
# image = cv2.hconcat([image1, image2])
# print(type(image))
# st.image(image, caption='Side by Side Image')
