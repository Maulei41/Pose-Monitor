import cv2
import numpy as np

# Demonstrates a class called Draw_predict that is used to draw keypoints and connections on an image using OpenCV
class Draw_predict(object):
    def __init__(self):
        self.edge =  {
            (0, 1): 'm',
            (0, 2): 'c',
            (1, 3): 'm',
            (2, 4): 'c',
            (5, 7): 'm',
            (7, 9): 'm',
            (6, 8): 'c',
            (8, 10): 'c',
            (5, 6): 'y',
            (5, 11): 'm',
            (6, 12): 'c',
            (11, 12): 'y',
            (11, 13): 'm',
            (13, 15): 'm',
            (12, 14): 'c',
            (14, 16): 'c'
        }
    # The draw_keypoints method takes an image, a set of keypoints, and a threshold value as input.
    # It draws circles at the locations of keypoints that have a confidence value above the threshold.
    def draw_keypoints(self, frame, keypoints, th1):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > th1:
                cv2.circle(frame, (int(kx), int(ky)), 3, (255, 255, 255), -1)
    # The draw_connections method takes the same inputs and draws lines between keypoints based on predefined connections.
    def draw_connections(self, frame, keypoints, th1):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for edge, color in self.edge.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            if (c1 > th1) & (c2 > th1):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
