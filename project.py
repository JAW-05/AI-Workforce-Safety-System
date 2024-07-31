import streamlit as st
import PIL
import cv2
import numpy as np
import utility
import io

def play_video(video_source):
    camera = cv2.VideoCapture(video_source)

    st_frame = st.empty()
    while(camera.isOpened()):
        ret, frame = camera.read()

        if ret:
            visualized_image = utility.predict_image(frame, conf_threshold = conf_threshold)
            st_frame.image(visualized_image, channels = "BGR")

        else:
            camera.release()
            break

st.set_page_config(
    page_title="AI Workforce Safety System",
    page_icon=":construction_man:",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("AI Workforce Safety Systemt :construction_man::construction_sign:")

st.sidebar.header("Type of PPE Detection")
source_radio = st.sidebar.radio("Select Source", ["IMAGE", "VIDEO", "WEBCAM"])

st.sidebar.header("Confidence")
conf_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 10, 100, 20))/100

input = None
if source_radio == "IMAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image", type = ("jpg", "png"))

    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        visualized_image = utility.predict_image(uploaded_image_cv, conf_threshold = conf_threshold)

        
        st.image(visualized_image, channels = "BGR")

    else:
        st.image("assets/thumbnail.jpg")
        st.write("Click on 'Browse Files' in the sidebar to run inference on an image.")

temporary_location = None
if source_radio == "VIDEO":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose a video", type = ("mp4"))

    if input is not None:
        g = io.BytesIO(input.read())
        temporary_location = "upload.mp4"

        with open(temporary_location, "wb") as out:
            out.write(g.read())

        out.close()

    if temporary_location is not None:
        play_video(temporary_location)

    else:
        st.video("assets/thumbnail.jpg")
        st.write("Click on 'Browse Files' in the sidebar to run inference on a video.")

if source_radio == "WEBCAM":
        play_video(0)
