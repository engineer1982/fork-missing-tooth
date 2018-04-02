# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 12:51:13 2018

@author: Marcelo Mota de Azevedo Junior
"""

import numpy as np
from PIL import Image
import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

def load_images():
    image_list = []
    label_list = []
    
    #folders = ["bkg", "full", "1", "2", "3", "4", "5", "6"]
    folders = ["bkg", "full", "6"]
    
    size = len(folders)
    
    for k in range(0,size):
        
        print("varrendo folder " + folders[k])
    
        for filename in glob.glob('data/train/' + folders[k] + '/*.jpg'): #assuming jpg  
            
            label = np.zeros((size))
            img = load_img(str(filename))  # this is a PIL image
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            image_list.append(x)
            label[k] = 1
            label_list.append(label)
    
    print("vai começar a converter")
    imagem = np.array(image_list)
    print("imagem convertida em np.array")
    labels = np.array(label_list)
    print("labels convertidos em np.array")
    
    return imagem, labels

batch_size = 32

epochs = 3

# bloco: COnvNets

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# bloco Fully Connected

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

x_train, y_train = load_images()

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

