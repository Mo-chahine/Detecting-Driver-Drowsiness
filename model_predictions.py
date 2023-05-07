import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from keras import layers, models
import tensorflow as tf
from sklearn.model_selection import train_test_split
tf.config.optimizer.set_jit(False)


EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
TEST_SIZE = 0.2


def load_data(subdirs):
    images = []
    labels = []
    # getting the list of all the subdirectories within the data folder each of which is a label for our model
    for subdir in subdirs:
        for filename in os.listdir(os.path.join("", str(subdir))):
            # loading the image file using opencv module
            img = cv2.imread(subdir+"/"+filename)

            # resizing the image matrix to make suitable for feeding into neural net
            resized_img = cv2.resize(
                img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

            gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            # storing the image array into a list
            images.append(gray_image)

            # as each subdir name indicates a category of sign so it's our label
            labels.append(subdir)

    return (images, labels)


def pixel_density(images):
    result = []
    for image in images:
        height, width = image.shape

        # Get the image resolution (pixels per inch)
        x_resolution, y_resolution = cv2.getOptimalDFTSize(
            width), cv2.getOptimalDFTSize(height)

        # Get the physical dimensions of the image (in inches)
        x_inches = width / x_resolution
        y_inches = height / y_resolution

        # Calculate the pixel density (dpi)
        dpi = (x_resolution, y_resolution)
        result.append(dpi)
    return result


def get_model():
    # Create the model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


subdirs = ["closed_eyes", "open_eyes"]
subdirs2 = ["yawn", "no_yawn"]
images, labels = load_data(subdirs)
images2, labels2 = load_data(subdirs2)

print("size of eyes dataset: ", len(labels))
print("size of mouths dataset: ", len(labels2))

for i in range(len(labels)):
    if labels[i] == "closed_eyes":
        labels[i] = 1
    else:
        labels[i] = 0
for i in range(len(labels2)):
    if labels2[i] == "yawn":
        labels2[i] = 1
    else:
        labels2[i] = 0


eye_images = np.array(images)
eye_labels = np.array(labels)
mouth_images = np.array(images2)
mouth_labels = np.array(labels2)
eye_images = eye_images.reshape(eye_images.shape[0], IMG_WIDTH, IMG_HEIGHT, 1)
mouth_images = mouth_images.reshape(
    mouth_images.shape[0], IMG_WIDTH, IMG_HEIGHT, 1)

eye_images = eye_images.astype('float32') / 255.0
mouth_images = mouth_images.astype('float32') / 255.0

x_train_eyes, x_test_eyes, y_train_eyes, y_test_eyes = train_test_split(
    eye_images, eye_labels, test_size=TEST_SIZE)

x_train_mouth, x_test_mouth, y_train_mouth, y_test_mouth = train_test_split(
    mouth_images, mouth_labels, test_size=0.1562)  # chose this test size to fix the size of the testing sample to be same length

# model for eyes
model = get_model()
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train_eyes, y_train_eyes, epochs=EPOCHS)
model.evaluate(x_test_eyes,  y_test_eyes, verbose=2)
ynew_eyes = model.predict(x_test_eyes)

# model for mouth
model2 = get_model()
model2.compile(optimizer='adam', loss='binary_crossentropy',
               metrics=['accuracy'])
model2.fit(x_train_mouth, y_train_mouth, epochs=EPOCHS)
model2.evaluate(x_test_mouth,  y_test_mouth, verbose=2)
ynew_mouth = model2.predict(x_test_mouth)

# changing the values of the predictions to 0 or 1 instead of probabilities
for i in range(len(ynew_mouth)):
    if ynew_eyes[i] >= 0.5:
        ynew_eyes[i] = 1
    else:
        ynew_eyes[i] = 0
    if ynew_mouth[i] >= 0.5:
        ynew_mouth[i] = 1
    else:
        ynew_mouth[i] = 0


# calculating how many wrong predictions in each
eyes_wrong = 0
mouths_wrong = 0
for i in range(len(ynew_mouth)):
    if ynew_mouth[i]-y_test_mouth[i] != 0:
        mouths_wrong += 1

    if ynew_eyes[i]-y_test_eyes[i] != 0:
        eyes_wrong += 1

print("Wrong mouth predictions= ", mouths_wrong,
      mouths_wrong*100/len(y_test_mouth))
print("Wrong eyes predictions= ", eyes_wrong, eyes_wrong*100/len(y_test_eyes))

print(ynew_eyes[200], y_test_eyes[200])
print(ynew_mouth[111], y_test_mouth[111])
