import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import time
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow
import sklearn 
import matplotlib
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

dataset_path = 'asl-dataset'

img_size = (400, 400) 

# Function: load and preprocess images
def load_and_preprocess_data(dataset_path, img_size):
    data = []
    labels = []

    start_time = time.time()
    # Loop through each folder in the dataset to generate labels for classification, then resize and normalize the images and add photo and label into an array
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        
        if not os.path.isdir(category_path):
            continue

        label = int(category) if category.isdigit() else ord(category.lower()) - ord('a') + 10
        print(label)

        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            
            try:
                img = cv2.imread(img_path)

                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                img = img / 255.0

                data.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error loading image: {img_path}, Error: {str(e)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to process images: {elapsed_time} seconds")

    return np.array(data), np.array(labels)

data, labels = load_and_preprocess_data(dataset_path, img_size)

num_classes=36

# one-hot encoding
labels_one_hot = to_categorical(labels, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)

#------------------------ CNN MODEL ----------------------------- #
startCNN = time.time()


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(400, 400, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5)) 
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 64 
epochs = 10
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Save the model
model.save('asl_model.h5')
endCNN = time.time()
totalCNN = endCNN - startCNN
print(f"Time taken to train model: {totalCNN} seconds")

print("DONE")
