import streamlit as st
import  cv2
st.title("Image Page")
with st.container():
    image1_placeholder, image2_placeholder = st.columns(2)
    image_upload1 = image1_placeholder.file_uploader("Upload Image 1", type=['jpg', 'png'])
    image_upload2 = image2_placeholder.file_uploader("Upload Image 2", type=['jpg', 'png']) 
    

    image1_placeholder.header("Image 1")
    image2_placeholder.header("Image 2")
    image1 = cv2.imread('../../image1_60_left.jpg')
    image2 = cv2.imread('../../image1_60_right.jpg')
    with image1_placeholder:
        st.image(image1, channels="BGR")
    with image2_placeholder:
        st.image(image2, channels="BGR")
# image = cv2.hconcat([image1, image2])
# print(type(image))
# st.image(image, caption='Side by Side Image')
