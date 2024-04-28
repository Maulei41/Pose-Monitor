import streamlit as st


st.set_page_config(page_title="Log Book", page_icon="For_ASS.jpeg", layout="centered")
col1, col2 = st.columns([1, 8])
col1.image("For_ASS.jpeg")
col2.title("Log Book")
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

logbook_data = [
    {
        "date": "14/9/2023",
        "time": "14:30-17:30",
        "content": """Generate the idea of project:

Use Movenet model to detect human posture key points and classify the standard posture 

-> What posture?
>Unknown, probably to classify the gym posture 
->What program language?
>  Python and Java
-> How many members now?
> Three, Angus Li, Alex Lau, Sunny Yau
-> What should we do now?
> Data Collect for the next meeting
"""
    },{
        "date": "5/10/2023",
        "time": "14:30-17:30",
        "content": """Added a new member
- Sunny Chan
Discuss the feasibility of the project:
> Created the group name “∀ ASS” and the team logo
-> Share the data collected a few weeks ago
> the project will become two part: computer vision and machine learning part -> computer vision part include get frame and process it in Movenet; machine learning part include pose classification model
> Movenet model downloads link
> Python library may be used (OpenCV, TensorFlow) -> How to use library->learn pip install library -> learn OpenCV and tensorflow commands through website and YouTube 
> Discuss the target user-> static fitness posture user；people who do have time go to the gym room
> Classified what posture -> static fitness postur, such as plank. 
> Decide two kinds of Movenet model: thunder and lightning -> let user decide to use what model they need
-> What should we do now?
> Do the demo of computer vision part now for next meeting
"""
    },{
        "date": "19/10/2023",
        "time": "14:30-17:30",
        "content": """ > Done the demo through OpenCV and TensorFlow
 > Discuss the unique selling points
 > Discuss about the Python programming code-> only work in
 local device
 > Discuss the UI interface of the program-> the function of UI
 interface had been discussed
"""
    },{
        "date": "31/10/2023",
        "time": "14:30-17:30",
        "content": """ ＞discover an app”SmartRehab” developed by HKU team and Hong Kong Society for Rehabilitation
＞we are afraid that our idea and project will have plagiarism, discuss to change our project idea
＞Figure out that only the idea between “SmartRehab”and our project are similar , but the beneficiary and the focus are different.
＞ We decided to stay on our original plan

"""
    },{
        "date": "2/11/2023",
        "time": "10:00-13:35",
        "content": """ ＞Try to make an interface of the demo by using tkinter-> fail
 >Discuss about using other method to replace it, Such as
 Streamlit, PyQt, PySise for UI-> Decide to use Streamlit to
 become a web UI interface for any device you can use
"""
    },{
        "date": "3/11/2023",
        "time": "22:00-23:45",
        "content": """ >Discuss about 2 version to do our apps
>Both of them base of python program language

>Version 1:
>Using OpenCV to deal with the image
>Using Movenet Model to getting human >node point
>Using algorithm and OpenCV to draw and direct 

>Version 2:
>Which is the plan that we are now using:
>Based on the original version , added a new UI interface, make it more user-friendly and beautify the appearance.
>Change OpenCV to streamlit webRTC for getting frame from Webcam 
>Optimize the project, reduce the burden of CPU and GPU by the self-built image cropping algorithm

>Finished project:
>OpenCV part:
．Use opencv to obtain and process images to input Movenet model 
．Draw points and lines in images with opencv
>Movenet part :
．Download the model
．Build up the interface between python   projects and movenet model
．Get the coordinates of 17 Human body key point in the movenet model

>Self-developed part:
．Build up the console menu interface
．Build up a menu to choose different mode

"""
    },{
        "date": "6/11/2023",
        "time": "21:00-24:00",
        "content": """  >Prepare our presentation 
>Edit our PowerPoint 

"""
    },{
        "date": "7/11/2023-8/11/2023",
        "time": "",
        "content": """  >Continue editing powerpoint
>Add the information of section 5 & 6
>Preparing the presentation scrip

"""
    },{
        "date": "9/11/2023",
        "time": "18:00-23:00",
        "content": """  ＞create ｢Pose-Detection.zip｣ to install all the resources and programs
＞put our software to the website for testing
> http://localhost:8501/（for testing, can not work now）
> only run on local->figure out reason : it need to change to https for other user use 

"""
    },{
        "date": "10/11/2023",
        "time": "12:00-15:00",
        "content": """  ＞deploy the demo on the computer and use ssl-proxy to redirect it to https-> only work in same local network -> port forwarding and domain is needed 
"""
    },{
        "date": "11/11/2023",
        "time": "13:00-17:00",
        "content": """   >deploy the program in Angus Li’s computer and use ddns-go and port forwarding to redirect the domain to local ip address 
> https://www.maulei41.com:8081 ( for testing, not work now)
> other user can successfully access into the domain ( not in same network )
＞only run on computer, can not run on phone(hardware efficacy issue)
＞figure out why it can not work on phone, the mobile phone camera pixels are too high, it need more time to process
＞we use figure( a toy ) to test our demo
＞find out it is better to set the confident threshold between 0.3-0.4
＞collect information about teachable machine, for future machine learning part.

"""
    },{
        "date": "23/11/2023",
        "time": "13:00-16:00",
        "content": """   ＞discuss the AI training model(use the model from the internet or develop a model by ourselves)
＞ two possible way: Use teachable to train the model, the model will be Javascript file; training the model ourselves through tensorflow. 
> we need to collect enough training data and test data now.

"""
    },{
        "date": "7/12/2023",
        "time": "15:00-18:00",
        "content": """   ＞edit log book
>Future Plan
The project the we need to complete in future:
．Build up a beautiful Webui interface with Streamlit
．Use Streamlit's own widget to add user changes to variable variables in Streamlit 
．Construct image cropping algorithm
．Collect data to train the model (training data set & standard posture data set）
．Build posture classifier model and change to .tflite file and the interface with python 

"""
    },{
        "date": "12/12/2023",
        "time": "20:00",
        "content": """   FinishedSem1logbook(ﾉ◕ヮ◕)ﾉ*:･ﾟ✧
"""
    },{
        "date": "23/12/2023",
        "time": "23:20",
        "content": """    Sem break Mission:
>Program for collecting human key
>Point to data files
>Program for creating and training model
>Model testing

"""
    },{
        "date": " 23/12/2023-20/01/2024",
        "time": "",
        "content": """     >Work on the Sem break Mission
"""
    },{
        "date": " 25/01/2024",
        "time": "14:00-17:00",
        "content": """ Add some new features:
>add a box on the first page of the website called “show more”
．when the user starts to use the app and picks the “show      more” box. the more detail data(keypoint name, X   coordinates,Y coordinates and confidence ) will be display

"""
    },{
        "date": " 02/02/02024 ",
        "time": "13:00-16:00",
        "content": """  Discussion
>Create side bar in the web app
．it provide the short cut to other pages , and display some    information
．The first page(Home):present how to use the project, project idea,example to the user
．The third page(Upload file)(Still need develop): for user to input the photo and get the keypoint message and the photo with keypoint
．The source page: choose different page and see the source code
>Still need problem to Solve
．Streamlit upload file function will return the byte object

"""
    },{
        "date": " 03/02/2024-07/02/2024 ",
        "time": "",
        "content": """ >Making the software which can able to use in https
>Testing the application can available in mobile phone or not
>Researching more standard and non-standard posture photo for later Ai training  
>Working on the interim report 
>Problem was asked by classmate After the presentation:
Q:Will it have any privacy problem(camera) when using the web app
Ans: the data will only save on the local device, won't shared to the server,so no privacy problem


"""
    },{
        "date": " 23/02/2024-01/03/2024 ",
        "time": "",
        "content": """ >Researching more photo in standard and non-standard data set
>Attempting to try using the classification model (Movenet and >TensorFlow lite pose classification mode,
>Debugging the classification model

"""
    },{
        "date": " 08/03/2024 ",
        "time": "13:00-16:00",
        "content": """ >Using the classification model 
>Training the standard and non-standard data set
>Divide into training and testing data set
>Succeeding the classify it is standard or not


"""
    },{
        "date": " 22/03/2024 ",
        "time": "13:00-16:00",
        "content": """ >Sampling the training data with Alex Lau, add it to the original data set 
>Retraining the classification model by using larger data set
>Learning the python time library
>Testing the application by using camera 
>Try to add the pop-up page function on the web app to create a guideline by using the Streamlit-modal(failed)
> add one more page for showing the guideline



"""
    },{
        "date": " 12/04/2024 ",
        "time": "13:00-16:00",
        "content": """ >Release the Final report, A2 poster, Sem 2 Logbook, Project deliverable
>Editing logbook

"""
    },{
        "date": " 12/04/2024 - 25/04/2024 ",
        "time": "",
        "content": """ >Working on the Final report, A2 poster, Sem 2 Logbook, Project deliverable
> recoding the user guideline video
>Take the presentation video and application Demo video

"""
    },{
        "date": " 26/04/2024 ",
        "time": "13:30",
        "content": """ >Presentation Day(13:30)
"""
    },
    # Add more meeting entries here
]

# Display the logbook
st.title("Project Logbook")

for entry in logbook_data:
    st.subheader(f"Date: {entry['date']}")
    st.subheader(f"Time: {entry['time']}")
    st.write(entry['content'])
    st.write("---") # Separator between entries
