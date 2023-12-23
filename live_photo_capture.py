import cv2
import mediapipe as mp
import numpy as np
import time

# Time Tracking
startCamera = time.time()



# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('ASL to Text Translator', frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    # Capture and process the close-up image when the spacebar is pressed
    if key == ord(' '):
        # Get the bounding box around the hand (full frame in this case)
        bbox = [0, 0, frame.shape[1], frame.shape[0]]
        
        # Extract the region of interest (ROI)
        hand_roi = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

        # Make the background black
        hand_roi[np.where((hand_roi == [0, 0, 0]).all(axis=2))] = [0, 0, 0]

        # Display the close-up image
        cv2.imshow('Close-Up Hand Image', hand_roi)

        # Save the close-up image with a black background
        cv2.imwrite('hand_closeup.png', hand_roi)

    # Break the loop when 'q' is pressed
    elif key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()




# Time Tracking
endCamera = time.time()
totalCamera = endCamera - startCamera
print(f"Time taken to take photo: {totalCamera} seconds")
print("DONE")