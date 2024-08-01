import streamlit as st
import PIL
import cv2
import numpy as np
import utility
import io

# Define user credentials
USER_CREDENTIALS = {
    "admin": "password123"
}

def play_video(video_source):
    camera = cv2.VideoCapture(video_source)
    st_frame = st.empty()
    while camera.isOpened():
        ret, frame = camera.read()
        if ret:
            try:
                visualized_image = utility.predict_image(frame, conf_threshold)
                st_frame.image(visualized_image, channels="BGR")
            except Exception as e:
                st.error(f"Error processing frame: {e}")
                break
        else:
            camera.release()
            break

def login():
    st.title("AI Workforce Safety System")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["logged_in"] = True
            st.success("Login successful")
        else:
            st.error("Invalid username or password")

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    st.set_page_config(
        page_title="AI Workforce Safety System",
        page_icon=":construction_worker:",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.title(":construction: AI Workforce Safety System :construction_worker:")

    st.sidebar.header("Type of PPE Detection")
    source_radio = st.sidebar.radio("Select Source", ["IMAGE", "VIDEO", "WEBCAM"])

    st.sidebar.header("Confidence")
    conf_threshold = float(st.sidebar.slider("Select the Confidence Threshold", 10, 100, 20)) / 100

    input = None
    if source_radio == "IMAGE":
        st.sidebar.header("Upload")
        input = st.sidebar.file_uploader("Choose an image", type=("jpg", "png"))

        if input is not None:
            try:
                uploaded_image = PIL.Image.open(input)
                uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
                visualized_image = utility.predict_image(uploaded_image_cv, conf_threshold)
                st.image(visualized_image, channels="BGR")
            except Exception as e:
                st.error(f"Error processing image: {e}")
        else:
            st.image("assets/thumbnail2.png")
            st.write("Click on 'Browse Files' in the sidebar to run inference on an image.")

    temporary_location = None
    if source_radio == "VIDEO":
        st.sidebar.header("Upload")
        input = st.sidebar.file_uploader("Choose a video", type=("mp4"))

        if input is not None:
            g = io.BytesIO(input.read())
            temporary_location = "upload.mp4"
            with open(temporary_location, "wb") as out:
                out.write(g.read())
            out.close()

        if temporary_location is not None:
            play_video(temporary_location)
        else:
            st.video("assets/thumbnail2.png")
            st.write("Click on 'Browse Files' in the sidebar to run inference on a video.")

    if source_radio == "WEBCAM":
        play_video(0)

    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.experimental_rerun()
