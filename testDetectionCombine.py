from rembg import remove
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model
import os
import time
import mediapipe as mp

def translatedResult(predicted_class):
    translate = {
        '0': 0,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
        '5': 5,
        '6': 6,
        '7': 7,
        '8': 8,
        '9': 9,
        '10': 'A',
        '11': 'B',
        '12': 'C',
        '13': 'D',
        '14': 'E',
        '15': 'F',
        '16': 'G',
        '17': 'H',
        '18': 'I',
        '19': 'J',
        '20': 'K',
        '21': 'L',
        '22': 'M',
        '23': 'N',
        '24': 'O',
        '25': 'P',
        '26': 'Q',
        '27': 'R',
        '28': 'S',
        '29': 'T',
        '30': 'U',
        '31': 'V',
        '32': 'W',
        '33': 'X',
        '34': 'Y',
        '35': 'Z'
    }
    print(f"Translated letter: {translate[str(predicted_class)]}")
    return translate[str(predicted_class)]

def remove_background_and_predict(input_path, output_path, model_path):
    # Remove background
    input_image = Image.open(input_path)
    output_image = remove(input_image)

    # Create a new image with a black background
    new_image = Image.new("RGBA", output_image.size, (0, 0, 0, 255))
    new_image.paste(output_image, (0, 0), output_image)

    # Save the final image with a black background
    new_image.save(output_path)

    # Load the trained model
    model = load_model(model_path)

    # Load the test image
    test_img = cv2.imread(output_path)
    if test_img is None:
        print(f"Error loading image from path: {output_path}")
        return None

    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(test_img, (400, 400))
    test_img = test_img / 255.0
    test_img = np.expand_dims(test_img, axis=0)

    # Make predictions
    predictions = model.predict(test_img)

    # Assuming it's a classification task, get the predicted class
    predicted_class = np.argmax(predictions)

    return predicted_class

def capture_hand_image():
    # Time Tracking
    startCamera = time.time()

    # Open the camera
    cap = cv2.VideoCapture(0)

    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)  # Set max_num_hands to 1

    # Variable to store the last detected hand landmarks
    last_hand_landmarks = None
    predicted_letters = []


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
        cv2.imshow('ASL to Text Translator', frame)

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
                cv2.imshow('Close-Up Hand Image', hand_roi)

                print("Image saved")

                input_path = 'hand_without_landmarks.png'
                output_path = 'hand_without_landmarks_clear.png'
                model_path = 'als_model.h5'

                # Call the background removal and prediction function
                predicted_class = remove_background_and_predict(input_path, output_path, model_path)

                if predicted_class is not None:
                    # print(f"Predicted Class: {predicted_class}")
                    letter = translatedResult(predicted_class)
                    predicted_letters.append(letter) # Terminal
                    


        # Break the loop when 'q' is pressed
        elif key == ord('q'):
            print("Predicted Letters:", ''.join(map(str, predicted_letters)))  # Terminal
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    
    
    # Time Tracking
    endCamera = time.time()
    totalCamera = endCamera - startCamera
    # print(f"Time taken to take photo: {totalCamera} seconds")
    print("DONE")







