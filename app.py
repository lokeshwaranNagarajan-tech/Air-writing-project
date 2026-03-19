import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np

# RTC Configuration (Google's STUN server for better connectivity)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # Image size-a kuraichu process panna fast-aa irukum (Resizing)
        h, w, _ = img.shape
        small_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        
        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        # Hand Tracking on smaller image
        img_rgb = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                lm8 = hand_lms.landmark[8]
                cx, cy = int(lm8.x * w), int(lm8.y * h)

                if self.prev_x == 0 and self.prev_y == 0:
                    self.prev_x, self.prev_y = cx, cy

                cv2.line(self.canvas, (self.prev_x, self.prev_y), (cx, cy), (0, 255, 0), 10)
                self.prev_x, self.prev_y = cx, cy
        else:
            self.prev_x, self.prev_y = 0, 0

        return cv2.addWeighted(img, 0.7, self.canvas, 1, 0)

st.title("Air Writing - Lite Version ⚡")
webrtc_streamer(
    key="air-writing-lite",
    video_transformer_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}, # Audio-va disable panna fast-aa irukum
)
