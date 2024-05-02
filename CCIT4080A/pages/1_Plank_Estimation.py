import sys
sys.path.append(r"..Pose-Monitor\CCIT4080A\ml")
sys.path.append(r"..\CCIT4080A")

import av
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import (webrtc_streamer, WebRtcMode)
from ml.Movenet import Movenet
from ml.Classifier import Classifier
from ml.Draw_predict import Draw_predict
from ml.Add_html import Add_html
import queue
import time

st.set_page_config(page_title="Plank Estimation", page_icon="For_ASS.jpeg", layout="centered")
col1, col2 = st.columns([1, 8])
col1.image("For_ASS.jpeg")
col2.title("Plank Estimation")
st.header("", divider="red")
with st.sidebar:
    st.image("For_ASS.jpeg")
    st.title("∀ ASS Team members")
    st.header("", divider="red")
    mem1, mem2, mem3, mem4 = st.columns([1,1,1,1])
    mem1.write("Angus Li")
    mem2.write("Alex Lau")
    mem3.write("Sunny Yau")
    mem4.write("Sunny Chen")
    st.header("", divider="red")

model_name = st.selectbox("Movenet Model Select:", (
    "Movenet lightning (float 16)",
    "Movenet thunder (float 16)",
    "Movenet lightning (int 8)",
    "Movenet thunder (int 8)"))
th1 = st.slider("confidence threshold", 0.0, 1.0, 0.35, 0.05)
st.caption("Suggest the confidence threshold should be setted between 0.3 to 0.4 to get the best result")
#Movenet: A machine learning model used for pose estimation.
movenet = Movenet(model_name)
#Classifier: Another machine learning model for classifying poses.
classify = Classifier("pose_classifier.tflite", "pose_labels.txt")
draw_predict = Draw_predict()
# Audio Feedback: Different audio cues are provided based on the user's performance during the exercise.
add_html = Add_html()
Error = add_html.autoplay_audio("Error.mp3")
Missing_Sound = add_html.autoplay_audio("Missing_sound.mp3")
Standard_Sound1 = add_html.autoplay_audio("Standard_Sound1.mp3")
Standard_Sound2 = add_html.autoplay_audio("Standard_Sound2.mp3")
Standard_Sound3 = add_html.autoplay_audio("Standard_Sound3.mp3")
Non_Standard_Sound1 = add_html.autoplay_audio("Non_Standard_Sound1.mp3")
Non_Standard_Sound2 = add_html.autoplay_audio("Non_Standard_Sound2.mp3")


KEYPOINT = ["nose", "left eye", "right eye", "left ear", "right ear", "left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right wrist", "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"]
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()
label_queue = queue.Queue()
output_queue = queue.Queue()
# Video Frame Callback:
# This section defines the video_frame_callback function, which is called for each video frame received.
# It converts the frame to an ndarray, performs pose detection using MoveNet, classifies the pose using a classification model, and stores the results in the respective queues.
# It also draws the detected keypoints on the frame.
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    keypoints_with_scores = movenet.movenet(image)
    x, y, c = image.shape
    pose_class_names, output = classify.classtify(keypoints_with_scores)
        maxConfidence = 0
    for i in range(len(output)):
        if output[i] > maxConfidence:
            maxConfidence = output[i]
            maxPos = i
    output_label =pose_class_names[maxPos]
    output_queue.put(output)
    label_queue.put(output_label)
    draw_predict.draw_connections(image, keypoints_with_scores, th1)
    draw_predict.draw_keypoints(image, keypoints_with_scores, th1)
    keypoints_with_scores = np.multiply(keypoints_with_scores, [x, y, 1])
    result_queue.put(keypoints_with_scores)
    #image = cv2.putText(image, output_label,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
    return av.VideoFrame.from_ndarray(image, format="bgr24")

# WebRTC Streaming: This section uses the webrtc_streamer function to set up a video streaming session.
# It specifies the video frame callback function, RTC configuration, and media stream constraints.
# It also handles the display of labels and other messages using the st module.
webRTC =webrtc_streamer(key="Pose Detection",
                mode=WebRtcMode.SENDRECV,
                video_frame_callback=video_frame_callback,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True)

label_predict = st.empty()
label_msg = st.empty()
keypoint_message = st.empty()
fps_message = st.empty()
non_Standard_counter = 0
Standard_counter = 0
missing_counter = 0
fps = 0
keypoint_num = 0
keypoint_score = 0
# Pose Classification and Output:
# This section processes the results from the queues and performs pose classification based on the keypoints detected.
# It keeps track of counters for standard and non-standard poses and triggers sound alerts based on predefined thresholds.
# It also handles missing poses and triggers a sound alert when a pose is missing for a certain duration.
if webRTC.state.playing:
    start_time = time.time()
    start = time.time()

    continue_time = 0
    counter = 0
    fps1_counter = 0
    fps2_counter = 0

    while True:
        label = label_queue.get()
        result = result_queue.get()
        result = np.squeeze(result)

        while keypoint_num < 17:
            keypoint_score += result[:, 2][keypoint_num]
            keypoint_num += 1
        keypoint_score = keypoint_score / 17
        if keypoint_score > th1:
            missing_counter = 0
            if label == "non_standard":
                non_Standard_counter += 1
                Standard_counter = 0
                label_msg.write(f"Non Standrad Counter ={non_Standard_counter}")
                if non_Standard_counter == 90:
                    st.markdown(Non_Standard_Sound1, unsafe_allow_html=True)
                elif non_Standard_counter == 250:
                    st.markdown(Non_Standard_Sound2, unsafe_allow_html=True)
            elif label == "standard":
                Standard_counter += 1
                non_Standard_counter = 0
                label_msg.write(f"Standrad Counter = {Standard_counter}")
                if Standard_counter == 30:
                    st.markdown(Standard_Sound1, unsafe_allow_html=True)
                elif Standard_counter == 90:
                    st.markdown(Standard_Sound2, unsafe_allow_html=True)
                elif Standard_counter == 150:
                    st.markdown(Standard_Sound3, unsafe_allow_html=True)
        else:
            missing_counter += 1
            if missing_counter < 90:
                label_msg.write(f"Missing Counter = {missing_counter} ")
            elif missing_counter == 90:
                Standard_counter = 0
                non_Standard_counter = 0
                label_msg.write(f"Missing ")
                missing_counter = 0
                st.markdown(Missing_Sound, unsafe_allow_html=True)


        # Output Display:
        # The code displays the pose classification results, including the predicted label, confidence scores, and keypoint coordinates.
        # It also calculates and displays the frames per second (FPS) for real-time performance monitoring.

        output = output_queue.get()
        counter += 1
        fps = counter // (time.time()-start_time)


        check_pose = pd.DataFrame({"Non Standard": [str(round(output[0] * 100, 2)) + "%"],
                                   "Standard": [str(round(output[1] * 100, 2)) + "%"]}, index=["prediction"])
        label_predict.table(check_pose)



        more_result = pd.DataFrame({
            "Keypoints": KEYPOINT,
            "X Coordinate": result[:, 1],
            "Y Coordinate": result[:, 0],
            "confidence threshold": result[:, 2]
        }, index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        keypoint_message.table(more_result)
        fps_message.write(f"Fps = {fps}")
        keypoint_num = 0
        counter = 0
        start_time = time.time()


# External References:
# The code includes markdown messages that provide references to the MoveNet model and related projects.

st.markdown("This demo uses a model and code from")
st.markdown("https://tfhub.dev/google/movenet/singlepose/lightning/4")
st.markdown("https://tfhub.dev/google/movenet/singlepose/thunder/4")
st.markdown("https://tensorflow.google.cn/lite/tutorials/pose_classification?hl=zh-cn")
st.markdown("Many thanks to the project")
