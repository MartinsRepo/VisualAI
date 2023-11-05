import cv2
#from matplotlib import pyplot as plt
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

image = cv2.imread("distracted.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = hands.process(image)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

cv2.imshow('Image', image)
cv2.waitKey(0)

