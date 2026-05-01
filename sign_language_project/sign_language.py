import cv2
import mediapipe as mp
import os
import urllib.request

# Download hand landmarker model if not present
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

# Initialize mediapipe hand landmarker
base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)
mp_draw = mp.tasks.vision.drawing_utils

cap = cv2.VideoCapture(0)

def fingers_up(landmarks):
    fingers = []
    # thumb
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # fingers
    tips = [8, 12, 16, 20]
    bases = [6, 10, 14, 18]
    for tip, base in zip(tips, bases):
        if landmarks[tip].y < landmarks[base].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def recognize_letter(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return 'a'
    elif fingers == [0, 1, 1, 0, 0]:
        return 'b'
    elif fingers == [0, 1, 1, 1, 0]:
        return 'c'
    elif fingers == [0, 1, 1, 1, 1]:
        return 'd'
    elif fingers == [0, 1, 0, 0, 0]:
        return 'e'
    elif fingers == [1, 1, 0, 0, 0]:
        return 'f'
    elif fingers == [0, 1, 0, 0, 1]:
        return 'g'
    elif fingers == [0, 1, 0, 1, 0]:
        return 'h'
    elif fingers == [0, 0, 0, 0, 1]:
        return 'i'
    elif fingers == [0, 0, 0, 1, 1]:
        return 'j'
    elif fingers == [1, 0, 0, 0, 0]:
        return 'k'
    elif fingers == [1, 1, 1, 0, 0]:
        return 'l'
    elif fingers == [1, 1, 1, 1, 0]:
        return 'm'
    elif fingers == [1, 1, 1, 1, 1]:
        return 'n'
    elif fingers == [0, 1, 0, 0, 1]:
        return 'o'
    elif fingers == [1, 1, 0, 0, 1]:
        return 'p'
    elif fingers == [1, 1, 0, 1, 1]:
        return 'q'
    elif fingers == [1, 0, 0, 0, 1]:
        return 'r'
    elif fingers == [0, 0, 1, 0, 0]:
        return 's'
    elif fingers == [0, 0, 1, 0, 1]:
        return 't'
    elif fingers == [0, 0, 1, 1, 0]:
        return 'u'
    elif fingers == [0, 0, 1, 1, 1]:
        return 'v'
    elif fingers == [0, 1, 1, 0, 1]:
        return 'w'
    elif fingers == [0, 1, 1, 1, 0]:
        return 'x'
    elif fingers == [1, 0, 0, 0, 0]:
        return 'y'
    elif fingers == [1, 0, 0, 1, 0]:
        return 'z'
    else:
        return '?'

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_blur = cv2.GaussianBlur(frame_rgb, (5, 5), 0)
    
    # Convert to mediapipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_blur)
    
    # Detect hand landmarks
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # Draw connections
            connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
            for connection in connections:
                start_idx = connection.start
                end_idx = connection.end
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    start_point = hand_landmarks[start_idx]
                    end_point = hand_landmarks[end_idx]
                    start_pos = (int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0]))
                    end_pos = (int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0]))
                    cv2.line(frame, start_pos, end_pos, (255, 0, 0), 2)
            
            # Calculate bounding box
            h, w, c = frame.shape
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
            
            fingers = fingers_up(hand_landmarks)
            letter = recognize_letter(fingers)
            cv2.putText(frame, letter, (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Sign Language Recognition', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('screenshot.png', frame)
        print('Screenshot saved')

cap.release()
cv2.destroyAllWindows()