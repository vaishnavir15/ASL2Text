from rembg import remove
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model
import os

import mediapipe as mp
import tkinter as tk
from prediction import predict_result
from remove_background import remove_background
from translate_result import translate_result



def update_labels(letter_label, all_letters_label, letter, predicted_letters):
    letter_label.config(text=f"Current Letter: {letter}")

    all_letters_label.config(
        text=f"Predicted Letters: {' '.join(map(str, predicted_letters))}")


# Open the camera and initialize mediapipe hands, and store history of the hands for processing
def capture_hand_image():

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)

    last_hand_landmarks = None
    predicted_letters = []

    root = tk.Tk()
    root.title("ASL to Text Translator")

    predicted_letter_label = tk.Label(root, text="Current Letter: ", font=("Helvetica", 16))
    predicted_letter_label.grid(row=0, column=0, sticky="nw")

    all_letters_label = tk.Label(root, text="Predicted Letters: ", font=("Helvetica", 16))
    all_letters_label.grid(row=1, column=0, sticky="nw")

    def destroy_root():
        root.destroy()

    # Capture frame-by-frame, convert the BGR image to RGBB, create a copy of the frame without landmarks from mediapipe hands and display
    while True:
        ret, frame = cap.read()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_without_landmarks = frame.copy()

        results = hands.process(rgb_frame)

        #Draw and store landmarks on detected hand
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                last_hand_landmarks = hand_landmarks

        cv2.imshow('ASL to Text Translator (RIGHT HAND ONLY)', frame)

        key = cv2.waitKey(1)

        # Capture and save the close-up image when the spacebar is pressed
        if key == 32:
            print("Spacebar pressed")
            if last_hand_landmarks is not None:
                # Calculate a box based off the landmarks
                x = int(min(last_hand_landmarks.landmark,
                        key=lambda l: l.x).x * frame.shape[1])
                y = int(min(last_hand_landmarks.landmark,
                        key=lambda l: l.y).y * frame.shape[0])
                w = int((max(last_hand_landmarks.landmark, key=lambda l: l.x).x -
                        min(last_hand_landmarks.landmark, key=lambda l: l.x).x) * frame.shape[1])
                h = int((max(last_hand_landmarks.landmark, key=lambda l: l.y).y -
                        min(last_hand_landmarks.landmark, key=lambda l: l.y).y) * frame.shape[0])
                padding = 30
                side_length = max(w, h) + padding
                x = max(0, x - (side_length - w) // 2)
                y = max(0, y - (side_length - h) // 2)
                x = min(frame.shape[1] - side_length, x)
                y = min(frame.shape[0] - side_length, y)

                hand_roi = frame_without_landmarks[y:y +
                                                   side_length, x:x + side_length]

                cv2.imwrite('hand_without_landmarks.png', hand_roi)

                # paths for image of hands with black background, model
                input_path = 'hand_without_landmarks.png'
                output_path = 'hand_without_landmarks_clear.png'
                model_path = 'asl_model.h5'

                remove_background(input_path, output_path)
                predicted_class = predict_result(output_path, model_path)

                if predicted_class is not None:
                    letter = translate_result(predicted_class)
                    predicted_letters.append(letter) 
                    update_labels(predicted_letter_label,
                                  all_letters_label, letter, predicted_letters)

        # Break the loop when 'q' is pressed
        elif key == ord('q'):
            print("Predicted Letters:", ''.join(map(str, predicted_letters))) 
            update_labels(predicted_letter_label, all_letters_label, "")
            root.after(30000, destroy_root)
            break

        root.update_idletasks()
        root.update()

    cap.release()
    cv2.destroyAllWindows()

    