import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np

# --- SETTINGS ---
# Defining colors (BGR format)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)] # Blue, Green, Red, Black (Eraser)
color_index = 0 

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.canvas = None
        self.prev_x, self.prev_y = 0, 0
        self.current_color = (0, 255, 0) # Default Green
        self.brush_thickness = 5

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        # 1. DRAW BUTTONS ON SCREEN
        cv2.rectangle(img, (20, 10), (120, 60), (255, 0, 0), -1) # Blue
        cv2.rectangle(img, (140, 10), (240, 60), (0, 255, 0), -1) # Green
        cv2.rectangle(img, (260, 10), (360, 60), (0, 0, 255), -1) # Red
        cv2.rectangle(img, (380, 10), (520, 60), (255, 255, 255), -1) # Eraser Button
        cv2.putText(img, "ERASER", (400, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        # 2. HAND TRACKING
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                lm8 = hand_lms.landmark[8]
                h, w, c = img.shape
                cx, cy = int(lm8.x * w), int(lm8.y * h)

                # 3. COLOR SELECTION LOGIC
                if cy < 70: # If finger is in the button area
                    if 20 < cx < 120: self.current_color = (255, 0, 0); self.brush_thickness = 5
                    elif 140 < cx < 240: self.current_color = (0, 255, 0); self.brush_thickness = 5
                    elif 260 < cx < 360: self.current_color = (0, 0, 255); self.brush_thickness = 5
                    elif 380 < cx < 520: self.current_color = (0, 0, 0); self.brush_thickness = 30 # Thick Eraser
                
                # 4. DRAWING LOGIC
                else:
                    if self.prev_x != 0:
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (cx, cy), self.current_color, self.brush_thickness)
                    self.prev_x, self.prev_y = cx, cy
        else:
            self.prev_x, self.prev_y = 0, 0

        combined = cv2.addWeighted(img, 0.7, self.canvas, 1, 0)
        return combined

# STREAMLIT UI
st.title("Air Writing Pro 🎨")
webrtc_streamer(key="air-writing", video_transformer_factory=VideoProcessor)