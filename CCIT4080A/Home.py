import sys
sys.path.append(r"..\CCIT4080A")
import cv2
import streamlit as st
from ml import Classifier, Draw_predict
from ml import Movenet
import tensorflow as tf
import pandas as pd

st.set_page_config(page_title="Pose-Monitor", page_icon="For_ASS.jpeg", layout="centered")
col1, col2 = st.columns([1, 8])
col1.image("For_ASS.jpeg")
col2.title("Pose Monitor")
st.header("", divider="red")



with st.sidebar:
    st.image("For_ASS.jpeg")
    st.title("∀ ASS Team members")
    st.header("", divider="red")
    mem1, mem2, mem3, mem4 = st.columns([1, 1, 1, 1])
    mem1.write("Angus Li")
    mem2.write("Alex Lau")
    mem3.write("Sunny Yau")
    mem4.write("Sunny Chen")
    st.header("", divider="red")


st.subheader("Hello! We are team _∀ ASS_ and students from HKU SPACE!")
st.write("Here is some example of our pose classification project.")

st.header("Plank Pose", divider="red")
col3, col4 = st.columns([1, 1])


model_name = "Movenet thunder (int 8)"
movenet = Movenet.Movenet(model_name)
classify = Classifier.Classifier("pose_classifier.tflite", "pose_labels.txt")
draw_predict = Draw_predict.Draw_predict()

# The run_detect function processes input images, detects poses, classifies them, and returns the display image, pose label, and classification output.
def run_detect(input_image_path):
    display_image = cv2.imread(input_image_path)
    if display_image is None:
            raise FileNotFoundError("Image not found.")
    input_image = tf.io.read_file(input_image_path)
    input_image = tf.image.decode_jpeg(input_image)
    keypoints_with_scores = movenet.movenet(input_image)
    pose_class_names, output = classify.classtify(keypoints_with_scores)
    maxConfidence = 0
    for i in range(len(output)):
        if output[i] > maxConfidence:
            maxConfidence = output[i]
            maxPos = i
    output_label =pose_class_names[maxPos]
    draw_predict.draw_connections(display_image, keypoints_with_scores, 0.3)
    draw_predict.draw_keypoints(display_image, keypoints_with_scores, 0.3)
    display_image = cv2.resize(display_image, (612, 408))
    return display_image, output_label, output
# Two sample images are processed using the run_detect function, and the results are displayed in a tabular format using Streamlit.
try:
    image1_path = "test/Standard1.jpg"
    display_image_1, display_label_1, output1 = run_detect(image1_path)
except FileNotFoundError as e:
    image1_path = "CCIT4080A/test/Standard1.jpg"
    display_image_1, display_label_1, output1 = run_detect(image1_path)
    col3.image(display_image_1, caption= display_label_1)
    predic1 = pd.DataFrame({"Non Standard" : [str(round(output1[0]*100, 2)) + "%"],
                       "Standard" : [str(round(output1[1]*100, 2)) + "%"]}, index= ["prediction"])
col3.table(predic1)
try:
    image2_path = "test/_NonStandard1.jpg"
    display_image_2, display_label_2 , output2 = run_detect(image2_path)
except FileNotFoundError as e:
    image2_path = "CCIT4080A/test/_NonStandard1.jpg"
    display_image_2, display_label_2 , output2 = run_detect(image2_path)
    col4.image(display_image_2, caption= display_label_2)
    predic2 = pd.DataFrame({"Non Standard" : [str(round(output2[0]*100, 2)) + "%"],
                       "Standard" : [str(round(output2[1]*100, 2)) + "%"]}, index= ["prediction"])
col4.table(predic2)
st.header("", divider="red")
st.image("Poster.jpg")
st.header("", divider="red")
st.subheader("GitHub Source Code:")
st.write("https://pose-monitor.streamlit.app/")
