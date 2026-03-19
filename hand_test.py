import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Download the hand landmarker model if not present
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded.")

# Hand tracking settings using new API
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Webcam capture start
cap = cv2.VideoCapture(0)

print("Camera opening... Press 'q' to stop.")

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Detect hands
    detection_result = detector.detect(mp_image)

    # Draw landmarks if detected
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw connections
            vision.drawing_utils.draw_landmarks(
                img, 
                hand_landmarks, 
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                vision.drawing_styles.DrawingStyles.hand_connections_style(),
                vision.drawing_styles.DrawingStyles.hand_landmarks_style()
            )
            
            # Highlight index finger tip (landmark 8)
            h, w, c = img.shape
            index_tip = hand_landmarks[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Hand Tracking Test", img)

    # 'q' press panna window close aagum
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
