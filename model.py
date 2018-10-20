import csv
import cv2
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout, Activation, Reshape

# apply random brightness to the image
def random_gamma(image):
    gamma = random.uniform(0.3,2.5)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
 
    img = cv2.LUT(image, table)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    return img

def load_data(data_path):
    lines = []
    with open(data_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
    # Loading data
    images, measurements = [], []
    for line in lines:                
        image_center = cv2.cvtColor(cv2.imread(line[0].strip()), cv2.COLOR_BGR2RGB)
        image_left = cv2.cvtColor(cv2.imread(line[1].strip()), cv2.COLOR_BGR2RGB)
        image_right = cv2.cvtColor(cv2.imread(line[2].strip()), cv2.COLOR_BGR2RGB)       
        images.append(image_center)
        images.append(image_left)
        images.append(image_right)
        
        correction = 0.25
        steering_center = float(line[3])
        steering_left = steering_center + correction
        steering_right = steering_center - correction    
        measurements.append(steering_center)
        measurements.append(steering_left)
        measurements.append(steering_right)    
        
    return images, measurements

# perform data augmentation flipping and applying random brightness
def data_augmentation(images, measurements):
    augmented_images, augmented_measurements = [], []
    for i in range(len(images)):
        augmented_images.append(images[i])
        augmented_measurements.append(measurements[i])
        
        brightness_img = random_gamma(images[i])
        augmented_images.append(brightness_img)
        augmented_measurements.append(measurements[i])
        
        flip_image = cv2.flip(images[i], 1)
        augmented_images.append(flip_image)
        augmented_measurements.append(measurements[i]*-1.0)
        
        brightness_flip_img = random_gamma(flip_image)
        augmented_images.append(brightness_flip_img)
        augmented_measurements.append(measurements[i]*-1.0)
        
    return augmented_images, augmented_measurements

def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Reshape((66,200,3), input_shape=(65,320,3)))
    model.add(Convolution2D(24,5,5,subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


data_path = 'D:/Programming/Course Materials/Self-Driving Car Engineer/simulator-windows-64/'

# Loading data
images, measurements = load_data(data_path)

# Data augmentation
augmented_images, augmented_measurements = data_augmentation(images, measurements)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

model = build_model()
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, batch_size=32, verbose=1)
model.save('model.h5')