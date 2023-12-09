import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.utils import to_categorical


# categories for each letter/number
categories = {  0: "a",
                1: "b",
                2: "c",
                3: "d",
                4: "e",
                5: "f",
                6: "g",
                7: "h",
                8: "i",
                9: "j",
                10: "k",
                11: "l",
                12: "m",
                13: "n",
                14: "o",
                15: "p",
                16: "q",
                17: "r",
                18: "s",
                19: "t",
                20: "u",
                21: "v",
                22: "w",
                23: "x",
                24: "y",
                25: "z",
                26: "0",
                27: "1", 
                28: "2",
                29: "3",
                30: "4",
                31: "5",
                32: "6",
                33: "7",
                34: "8",
                35: "9"  
            }

# Set the path to your dataset
dataset_path = 'asl-dataset'

# Set the size of the images
img_size = (400, 400) 

# Function to load and preprocess images
def load_and_preprocess_data(dataset_path, img_size):
    data = []
    labels = []

    start_time = time.time()
    # Loop through each folder (0, 1, 2, ..., 9 and a, b, c, ..., z)
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        
        # Skip if it's not a directory
        if not os.path.isdir(category_path):
            continue

        # Get the label for the category
        label = int(category) if category.isdigit() else ord(category.lower()) - ord('a') + 10
        print(label)

        # Loop through each image in the category folder
        counter = 1
        for img_name in os.listdir(category_path):
            # print(counter)
            counter+=1
            img_path = os.path.join(category_path, img_name)
            
            try:
                # Read and resize the image
                img = cv2.imread(img_path)

                if img is None:
                    # Skip if the image couldn't be read
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                # Normalize pixel values to be between 0 and 1
                img = img / 255.0

                # Append the image and label to the data lists
                data.append(img)
                labels.append(label)
            except Exception as e:
                # Handle any exceptions that may occur during image loading
                print(f"Error loading image: {img_path}, Error: {str(e)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to process images: {elapsed_time} seconds")

    return np.array(data), np.array(labels)

# Load and preprocess the data

data, labels = load_and_preprocess_data(dataset_path, img_size)
print("Out of for loop, now will print num_classes ")
# Convert labels to one-hot encoding
num_classes = len(categories)
print(num_classes)
# labels_one_hot = to_categorical(labels, num_classes=num_classes)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)
