import streamlit as st
import  cv2
st.title("Image Page")
with st.container():
    image1_placeholder, image2_placeholder = st.columns(2)
    image1_placeholder.header("Image 1")
    image2_placeholder.header("Image 2")
    image1 = cv2.imread('../../image1_60_left.jpg')
    image2 = cv2.imread('../../image1_60_right.jpg')
image = cv2.hconcat([image1, image2])
st.image(image, caption='Side by Side Image')