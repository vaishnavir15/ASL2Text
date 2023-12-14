import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import time
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow
import sklearn  # Add this line to import the sklearn module
import matplotlib
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


print("OpenCV version:", cv2.__version__)
print("NumPy version:", np.__version__)
print("scikit-learn version:", sklearn.__version__)
print("Keras version:", tensorflow.__version__)
print("Matplotlib version:", matplotlib.__version__)

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
            print(counter)
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

num_classes=36

labels_one_hot = to_categorical(labels, num_classes=num_classes)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)
#----------> test_size = 0.2 -> 0.4 -> 0.1->0.2

# # Print shapes and types
# print("Shape of data:", data.shape)
# print("Shape of labels:", labels.shape)
# print("Data type of data:", data.dtype)

# # Visual Inspection: Display a few images
# for i in range(0, 2515, 60):
#     plt.imshow(data[i])
#     plt.title(f"Label: {labels[i]}")
#     plt.show()

# # Check the first few labels
# print("Sample labels:", labels[:5])

# # Check one-hot encoding
# print("One-hot encoded labels shape:", labels_one_hot.shape)

# # Check class distribution
# print("Class distribution in labels:", np.sum(labels_one_hot, axis=0))


# ------------------------ CNN MODEL ----------------------------- #
startCNN = time.time()
# Define the CNN model
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(400, 400, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(400, 400, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))  # Additional Conv2D layer
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu'))  # Additional Conv2D layer LETS TRY W THIS (need to beat 96.6)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))  # Increase Dense units 512->1024
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))






# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 10
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
#---------->validation_split = 0.1 -> 0.2 -> 0.3 -> 0.2-> 0.1 (bc of a tut)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Save the model
# model.save('asl_model.h5')
endCNN = time.time()
totalCNN = endCNN - startCNN
print(f"Time taken to train model: {totalCNN} seconds")


print("DONE")

