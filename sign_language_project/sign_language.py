import cv2
import mediapipe as mp
import urllib.request
import os

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )

base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

cap = cv2.VideoCapture(0)
timestamp = 0

def fingers_up(lm):
    fingers = []
    if lm[4].x < lm[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    tips = [8, 12, 16, 20]
    bases = [6, 10, 14, 18]
    for tip, base in zip(tips, bases):
        if lm[tip].y < lm[base].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def recognize_letter(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return 'a'
    elif fingers == [0, 1, 1, 1, 1]:
        return 'b'
    elif fingers == [0, 1, 1, 1, 0]:
        return 'c'
    elif fingers == [0, 1, 1, 0, 0]:
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
    elif fingers == [1, 1, 0, 0, 1]:
        return 'l'
    elif fingers == [1, 1, 1, 0, 0]:
        return 'm'
    elif fingers == [1, 1, 1, 1, 0]:
        return 'n'
    elif fingers == [1, 1, 1, 1, 1]:
        return 'o'
    elif fingers == [1, 0, 1, 0, 0]:
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
    elif fingers == [0, 0, 0, 1, 0]:
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

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_blur)
    result = detector.detect_for_video(mp_image, timestamp)
    timestamp += 1

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            h, w, _ = frame.shape

            x_min = w
            y_min = h
            x_max = 0
            y_max = 0

            coords = []
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                coords.append((cx, cy))
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                if cx < x_min: x_min = cx
                if cy < y_min: y_min = cy
                if cx > x_max: x_max = cx
                if cy > y_max: y_max = cy

            for start, end in connections:
                cv2.line(frame, coords[start], coords[end], (255, 0, 0), 2)

            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

            fingers = fingers_up(hand)
            letter = recognize_letter(fingers)
            cv2.putText(frame, letter, (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Sign Language Recognition', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('screenshot.png', frame)

cap.release()
cv2.destroyAllWindows()
