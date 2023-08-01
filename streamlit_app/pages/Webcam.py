import streamlit as st
import cv2
import numpy as np
from streamlit_extras.switch_page_button import switch_page

st.title("Webcam Page")
back_button  = st.button("Back to Home Page")
if back_button:
    switch_page("streamlit_app")

