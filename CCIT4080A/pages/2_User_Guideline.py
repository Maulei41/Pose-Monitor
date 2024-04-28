import streamlit as st

st.set_page_config(page_title="User Guideline", page_icon="For_ASS.jpeg", layout="centered")
col1, col2 = st.columns([1, 8])
col1.image("For_ASS.jpeg")
col2.title("User Guideline")
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


st.subheader("Here is the user guideline video")
st.video('user_guideline.mp4')

st.title("Plank Detector Instructions")

st.header("Selecting the MoveNet Model")
st.write("""
Navigate to the ‘Movenet Model Select’ dropdown menu. 

You can choose between 'Movenet lightning (float 16)', 'Movenet lightning (int 8)', 'Movenet thunder (float 16)', or 'Movenet thunder (int 8)'. 

The 'lightning' models are optimized for speed, while 'thunder' models are optimized for accuracy. 

We suggest starting with 'Movenet lightning (float 16)' for a balance between speed and accuracy.
""")

st.header("Adjusting the Confidence Threshold")
st.write("""
Adjust the 'Confidence Threshold' slider to set the sensitivity of the posture detection. 

This will determine how strictly the software judges your plank form. 

For the best results, we suggest setting the confidence threshold between 0.3 to 0.4. 

This range offers a balance, providing accurate feedback without being overly sensitive to minor deviations.
""")
st.header("allow Camera Access ")
st.write("""
Grant Camera Permission: 

Upon arriving at the website, a prompt will appear asking for permission to access your camera. 

Click ‘Allow’ to proceed. 

This step is crucial as the application uses your camera to capture video for posture analysis. 

""")

st.header("Position Yourself  ")
st.write("""
Set Up Your Space: 

Ensure you are in a well-lit area where your whole body can be seen by the camera. 

The camera should be placed at a distance where your entire body is visible, from head to toe. 

This helps in accurately capturing your posture during the plank exercise. 

Align the Camera: 

Adjust the camera so it is level with your body to provide a clear, straight-on view of your posture. 

This might require placing your device on a stable surface or using a tripod. 

""")

st.header("Start the Detection  ")
st.write("""
Initiate Posture Detection: 

Click the ‘Start’ button on the website. 

This will start the camera and begin analyzing your position before you start performing the plank.

""")

st.header("Perform the Plank ")
st.write("""
Get into Position: 

Start the plank exercise by positioning your forearms on the ground. 

Ensure that your elbows are directly below your shoulders and that your arms are parallel to your body, about shoulder-width apart. 

Maintain Proper Form: 

Keep your back straight, your buttocks in line with your shoulders, and your head facing down. 

It is crucial to maintain a neutral spine to avoid strain. 

""")
st.header("Prediction Accuracy ")
st.write("""
Non-Standard vs. Standard vs. Missing: 

After you start your session, the software will provide a real-time prediction, indicating the accuracy of your plank. 

A high "Standard" percentage indicates that your plank closely aligns with the ideal form. 

Conversely, a high "Non-Standard" percentage suggests that adjustments are needed for correct posture. 

If you are not in the camera, it will have a show Missing and have a Missing sound effect. 

""")

st.header("Key Points Detection ")
st.write("""
X and Y Coordinates: 

The application displays the detected key points of your body as coordinates on the screen. 

The X coordinate represents the horizontal axis, while the Y coordinate represents the vertical axis of the key point's position as captured by the camera. 

Each key point corresponds to specific parts of your body, such as your nose, left eye, right shoulder, etc. 

""")

st.header("Confidence Threshold  ")
st.write("""
Per Key Point:
 
Next to each key point, you will find a 'confidence threshold' score, showing how confident the model is that it has correctly detected and positioned that key point. 

A higher score signifies greater confidence. 

For example, if the 'left shoulder' key point has a high confidence threshold score, it means the model is quite certain it has located your left shoulder accurately. 

If the score is low, the model is less certain, and it may indicate that you are positioning in relation to the camera needs adjusting or that your form is incorrect

""")

st.header("Posture Analysis ")
st.write("""
Real-Time Feedback: As you hold the plank, the website will analyze your posture in real-time. 

It checks for alignment and balance, ensuring that your form meets the standard for a proper plank. 

Review the Feedback 

""")

st.header("End Session")
st.write("""
Conclude Your Exercise: 

When you are finished, click the ‘End’ button to stop the detection process. 
]
You can review a summary of your session and see tips for improvement. 
""")

st.header("Additional Tips ")
st.write("""
Wear Appropriate Attire: 

Choose tight-fitting clothing to allow the software to accurately detect your body’s outline and movements. 

Consistent Practice: 

Regular sessions with the Pose Detector can help improve your form over time, making the exercise more effective. 

Safety First: 

Always consult with a fitness professional or healthcare provider if you are unsure about your ability to perform exercise safely, especially if you have pre-existing health conditions
""")
