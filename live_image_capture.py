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


def capture_hand_image():

    # Open the camera
    cap = cv2.VideoCapture(0)

    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)  # Set max_num_hands to 1

    # Variable to store the last detected hand landmarks
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

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a copy of the frame without landmarks
        frame_without_landmarks = frame.copy()

        # Process the frame with Mediapipe Hands
        results = hands.process(rgb_frame)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Store the detected hand landmarks
                last_hand_landmarks = hand_landmarks

        # Display the resulting frame with landmarks
        cv2.imshow('ASL to Text Translator (RIGHT HAND ONLY)', frame)

        # Check for key press
        key = cv2.waitKey(1)

        # Capture and save the close-up image when the spacebar is pressed
        if key == 32:  # 32 is the ASCII code for spacebar
            print("Spacebar pressed")
            if last_hand_landmarks is not None:
                # Calculate bounding box based on landmarks
                x = int(min(last_hand_landmarks.landmark,
                        key=lambda l: l.x).x * frame.shape[1])
                y = int(min(last_hand_landmarks.landmark,
                        key=lambda l: l.y).y * frame.shape[0])
                w = int((max(last_hand_landmarks.landmark, key=lambda l: l.x).x -
                        min(last_hand_landmarks.landmark, key=lambda l: l.x).x) * frame.shape[1])
                h = int((max(last_hand_landmarks.landmark, key=lambda l: l.y).y -
                        min(last_hand_landmarks.landmark, key=lambda l: l.y).y) * frame.shape[0])

                # Add some padding and make it a square
                padding = 30
                side_length = max(w, h) + padding
                x = max(0, x - (side_length - w) // 2)
                y = max(0, y - (side_length - h) // 2)

                # Ensure the bounding box is within the image boundaries
                x = min(frame.shape[1] - side_length, x)
                y = min(frame.shape[0] - side_length, y)

                # Extract the region of interest (ROI) around the hand
                hand_roi = frame_without_landmarks[y:y +
                                                   side_length, x:x + side_length]

                # Save the close-up image without landmarks
                cv2.imwrite('hand_without_landmarks.png', hand_roi)

                # Display the close-up image without landmarks
                # cv2.imshow('Close-Up Hand Image', hand_roi)

                input_path = 'hand_without_landmarks.png'
                output_path = 'hand_without_landmarks_clear.png'
                model_path = 'asl_model.h5'

                # Call the background removal and prediction function
                remove_background(input_path, output_path)
                predicted_class = predict_result(output_path, model_path)

                if predicted_class is not None:
                    # print(f"Predicted Class: {predicted_class}")
                    letter = translate_result(predicted_class)
                    predicted_letters.append(letter)  # Terminal
                    update_labels(predicted_letter_label,
                                  all_letters_label, letter, predicted_letters)

        # Break the loop when 'q' is pressed
        elif key == ord('q'):
            print("Predicted Letters:", ''.join(
                map(str, predicted_letters)))  # Terminal
            update_labels(predicted_letter_label, all_letters_label, "")
            root.after(30000, destroy_root)
            break

        root.update_idletasks()
        root.update()

    # Release the camera and close all windows

    cap.release()
    cv2.destroyAllWindows()

    