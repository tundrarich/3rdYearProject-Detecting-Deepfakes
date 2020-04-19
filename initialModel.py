# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:22:03 2020

@author: Jay
"""

from keras import layers, models
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from keras.optimizers import Adam
from keras.models import Sequential
import keras.backend as K
from PIL import Image
from keras import layers, models
from keras.models import Sequential
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import modelStructure
import tensorflow as tf
import cv2
import glob
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
from keract import get_activations, display_activations

def initialModel():
    model = Sequential()
    
    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), strides = 1, padding='same', activation='relu', input_shape=(256,256,3)))
    model.add(layers.AveragePooling2D()) 
    
    model.add(layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D()) 
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(units=112, activation='relu'))
    model.add(layers.Dense(units=72, activation='relu'))
    model.add(layers.Dense(units=1, activation = 'sigmoid'))
    
    model.compile(optimizer=Adam(lr = 0.001),loss='binary_crossentropy', metrics=['accuracy'])   
  
    return model

def finalModel():
    model = Sequential()
    
    model.add(layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', input_shape=(256,256,3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))     

    model.add(layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))     
    
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same')) 
    model.add(layers.Flatten())

    model.add(layers.Dense(units=72, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(units=1, activation = 'sigmoid'))

    model.compile(optimizer=Adam(lr = 0.0008),loss='binary_crossentropy', metrics=['accuracy'])   
  
    return model

    
def main():
    batchSize = 100
    epochSize = 20
    
    trainingSize = 6973
    #validationSize = 3060
   
    
    model = finalModel()
    datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=60, zoom_range=0.2, rescale=1./255)
    
    trainGenerator = datagen.flow_from_directory('MixedData/Training/', class_mode='binary', batch_size=batchSize, shuffle=False)
    validationGenerator = datagen.flow_from_directory('MixedData/Validation/', class_mode='binary', batch_size=batchSize, shuffle=False)  
    print(validationGenerator.class_indices)
    
    model.fit_generator(trainGenerator, steps_per_epoch=trainingSize/batchSize, epochs=epochSize, validation_data=validationGenerator, initial_epoch=0)
   
    model_json = model.to_json()
    with open("Models/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("Models/model.h5")
    print("Saved model to disk")
    
   


def load():
    testingSize = 3042
    batchSize = 50
    
    json_file = open('Models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("Models/model.h5") 
    model.compile(optimizer = Adam(lr = 0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    print("Loaded model from disk")   
        
    
    
    datagen = ImageDataGenerator(rescale=1./255)
    testGenerator = datagen.flow_from_directory('Data/Deepfake/Testing/', class_mode='binary', batch_size=batchSize, shuffle=False)
    x,y = testGenerator.next()
    x,y = testGenerator.next()
    x,y = testGenerator.next()
    x,y = testGenerator.next()
    x,y = testGenerator.next()
    x,y = testGenerator.next()
    x,y = testGenerator.next()
    for i in range(0,1):
        image = x[1]
        plt.imshow(image)
        plt.show()
    image = image.reshape([1,256, 256,3])
    keractInputs = image
    #keract_targets = target_test[:1]
    activations = get_activations(model, keractInputs)
    display_activations(activations, save=False)
    
    results = model.evaluate(testGenerator, steps=testingSize/batchSize)
    
    print('test loss, test acc:', results)

   
    


        
main()
#load()
        