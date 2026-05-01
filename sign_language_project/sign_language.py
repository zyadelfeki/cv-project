import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def fingers_up(landmarks):
    fingers = []
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
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
    
    results = hands.process(frame_blur)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            h, w, c = frame.shape
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0
            for lm in hand_landmarks.landmark:
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
            
            fingers = fingers_up(hand_landmarks.landmark)
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