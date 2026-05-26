import cv2
import mediapipe as mp
import pyautogui
import time
import speech_recognition as sr
from collections import deque
import threading

# ===========================
# Select Mode at Startup
# ===========================
print("Choose Control Mode:")
print("1. Gesture Control")
print("2. Voice Control")
print("3. Both")
mode = input("Enter option (1/2/3): ")

use_gesture = mode in ['1', '3']
use_voice = mode in ['2', '3']

# ===========================
# Gesture Setup
# ===========================
if use_gesture:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    gesture_delay = 0.3
    last_action_time = 0
    last_gesture = "none"
    prev_frame_time = 0
    new_frame_time = 0
    gesture_buffer = deque(maxlen=5)

def get_finger_states(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    states = []
    for tip, pip in zip(finger_tips, finger_pips):
        states.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y else 0)
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_extended = 1 if thumb_tip.x > thumb_ip.x else 0
    return [thumb_extended] + states

def classify_gesture(hand_landmarks):
    fingers = get_finger_states(hand_landmarks)
    if fingers == [0, 1, 1, 0, 0]:
        return "hoverboard"
    elif fingers == [0, 1, 1, 1, 1]:
        return "jump"
    elif fingers == [0, 0, 0, 0, 0]:
        return "slide"
    else:
        index_tip_x = hand_landmarks.landmark[8].x
        thumb_tip_x = hand_landmarks.landmark[4].x
        if index_tip_x > thumb_tip_x + 0.1:
            return "left"
        elif index_tip_x < thumb_tip_x - 0.1:
            return "right"
        else:
            return "none"

def get_stable_gesture(new_gesture):
    gesture_buffer.append(new_gesture)
    if gesture_buffer.count(gesture_buffer[0]) == gesture_buffer.maxlen:
        return gesture_buffer[0]
    return "none"

def perform_action(gesture):
    global last_action_time, last_gesture
    current_time = time.time()
    if gesture != last_gesture or (current_time - last_action_time > gesture_delay):
        if gesture == "jump":
            pyautogui.press("up")
        elif gesture == "slide":
            pyautogui.press("down")
        elif gesture == "left":
            pyautogui.press("left")
        elif gesture == "right":
            pyautogui.press("right")
        elif gesture == "hoverboard":
            pyautogui.press("space")
            pyautogui.press("space")
        last_action_time = current_time
        last_gesture = gesture

# ===========================
# Voice Control Setup
# ===========================
if use_voice:
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    command_map = {
        "jump": "up",
        "slide": "down",
        "left": "left",
        "right": "right",
        "hoverboard": ["space", "space"]
    }

def voice_control_loop():
    while True:
        try:
            with mic as source:
                print("🎤 Listening for voice command...")
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=2)
            try:
                command = recognizer.recognize_google(audio).lower()
                print(f"🎙️ You said: {command}")
                for key in command_map:
                    if key in command:
                        action = command_map[key]
                        if isinstance(action, list):
                            for key_press in action:
                                pyautogui.press(key_press)
                                time.sleep(0.1)
                        else:
                            pyautogui.press(action)
                        print(f"✅ Performed: {key}")
                        break
                else:
                    print("❌ Unknown or unrecognized command")
            except sr.UnknownValueError:
                print("⚠️ Could not understand audio")
            except sr.RequestError as e:
                print(f"🔌 Speech Recognition error: {e}")
        except sr.WaitTimeoutError:
            print("⏳ No speech detected (timeout)")

if use_voice:
    threading.Thread(target=voice_control_loop, daemon=True).start()

# ===========================
# Main Gesture Loop
# ===========================
if use_gesture:
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        gesture = "none"

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                gesture = classify_gesture(handLms)
                mp_drawing.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

        stable_gesture = get_stable_gesture(gesture)
        perform_action(stable_gesture)

        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - prev_frame_time + 1e-5))
        prev_frame_time = new_frame_time

        cv2.putText(img, f"Gesture: {stable_gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"FPS: {fps}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Gesture Controlled Game", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    while True:
        time.sleep(1)  # Keep main thread alive for voice mode only
