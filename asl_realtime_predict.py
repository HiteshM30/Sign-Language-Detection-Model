import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# Load model
with open("asl_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Sentence building
sentence = ""
last_prediction = ""
gesture_locked = False
lock_time = 0
unlock_delay = 1.0  # seconds after hand disappears

def get_landmark_list(hand_landmarks):
    return [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    current_time = time.time()

    hand_detected = False
    prediction = ""

    if result.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = get_landmark_list(hand_landmarks)

            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]

                if not gesture_locked and prediction != last_prediction:
                    sentence += prediction
                    last_prediction = prediction
                    gesture_locked = True  # Lock after registering prediction
                    lock_time = current_time

                # Show predicted letter
                cv2.putText(frame, f'Letter: {prediction}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        # No hand detected
        if gesture_locked and current_time - lock_time > unlock_delay:
            gesture_locked = False
            last_prediction = ""

    # Show sentence
    cv2.putText(frame, f'Sentence: {sentence}', (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Instructions
    cv2.putText(frame, "Press [Space]=space  [Backspace]=del  [Enter]=save  [Esc]=exit",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("ASL Sentence Builder", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 13:  # ENTER
        print("Final Sentence:", sentence)
        with open("asl_sentence_output.txt", "a") as f:
            f.write(sentence + "\n")
        sentence = ""
    elif key == 8:  # Backspace
        sentence = sentence[:-1]
    elif key == 32:  # Spacebar
        sentence += " "

cap.release()
cv2.destroyAllWindows()
