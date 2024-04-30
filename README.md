# Pose-Monitor
![Poster](Poster.jpg)
##DEMO (https://pose-monitor.streamlit.app/)
## For deploy in server or in computer
please download the .zip file and extract it.

### Install
please make sure to use pip install the following lirbaries into your IDE:
```shell
pip install Pyav, streamlit, streamlit-webrtc, opencv-python-headless, pandas, numpy, tensorflow
```

### Run
please get into the file 'CCIT4080A'

for window user, you can use this command:
```shell
cd CCIT4080A
```

than, you can use these command to run the program in the Python IDE
```shell
streamlit run Home.py
```

### HTTPS

This program used this library "streamlit-webrtc"

refer to this [https://github.com/whitphx/streamlit-webrtc/tree/main]

`streamlit-webrtc` uses [`getUserMedia()`](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia) API to access local media devices, and this method does not work in an insecure context.

[This document](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia#privacy_and_security) says
> A secure context is, in short, a page loaded using HTTPS or the file:/// URL scheme, or a page loaded from localhost.

So, when hosting your app on a remote server, it must be served via HTTPS if your app is using webcam or microphone.
If not, you will encounter an error when starting using the device. For example, it's something like below on Chrome.
> Error: navigator.mediaDevices is undefined. It seems the current document is not loaded securely.

[Streamlit Community Cloud](https://streamlit.io/cloud) is a recommended way for HTTPS serving. You can easily deploy Streamlit apps with it, and most importantly for this topic, it serves the apps via HTTPS automatically by default.

For the development purpose, sometimes [`suyashkumar/ssl-proxy`](https://github.com/suyashkumar/ssl-proxy) is a convenient tool to serve your app via HTTPS.
```shell
$ streamlit run your_app.py  # Assume your app is running on http://localhost:8501
# Then, after downloading the binary from the GitHub page above to ./ssl-proxy,
$ ./ssl-proxy -from 0.0.0.0:8000 -to 127.0.0.1:8501  # Proxy the HTTP page from port 8501 to port 8000 via HTTPS
# Then access https://localhost:8000
