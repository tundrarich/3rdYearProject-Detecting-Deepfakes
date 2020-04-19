# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:59:53 2020

@author: Jay
"""
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
batchSize = 50

startEpoch = 11
epochSize = 20

learningRate = 0.001

trainingSize = 6973
validationSize = 2279
testingSize = 3035


classWeights = {0:1, 1:1}
#trainingSize = 7457
#validationSize = 2459
#testingSize = 2521
predictSize = 2



def trainInitial():
    
    model = modelStructure.MesoInception4()

     
    datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=60, zoom_range=0.2, rescale=1./255)

    
    
    
    trainGenerator = datagen.flow_from_directory('MixedData/Training/', class_mode='binary', batch_size=batchSize, shuffle=False)
    validationGenerator = datagen.flow_from_directory('MixedData/Validation/', class_mode='binary', batch_size=batchSize, shuffle=False)  
    print(validationGenerator.class_indices)
    
    model.fitGenerator(trainGenerator, validationGenerator, batchSize, epochSize, trainingSize, validationSize, classWeights)
    model.save()
    print("")
    
    
    file = open("Models/Log.txt","a")
    string = "Trained model initially with {} epochs. \n".format(epochSize)
    file.write(string) 
    file.close()

def loadModelContinueTraining():


    
    json_file = open('Models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("Models/model.h5") 
    model.compile(optimizer = Adam(lr = learningRate), loss = 'mean_squared_error', metrics = ['accuracy'])
    print("Loaded model from disk")
   
    datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=60, zoom_range=0.2, rescale=1./255)
    
      
    trainGenerator = datagen.flow_from_directory('MixedData/Training/', class_mode='binary', batch_size=batchSize, shuffle=False)
    validationGenerator = datagen.flow_from_directory('MixedData/Validation/', class_mode='binary', batch_size=batchSize, shuffle=False) 
  
    model.fit_generator(trainGenerator, steps_per_epoch=trainingSize/batchSize, epochs=epochSize, initial_epoch= startEpoch -1, validation_data=validationGenerator, 
                                       class_weight=classWeights)
    print("")
    

    model_json = model.to_json()
    with open("Models/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("Models/model.h5")
    print("Saved model to disk")
    
    
    file = open("Models/Log.txt","a")
    string = "Trained model from {} epochs to {} epochs. \n".format(startEpoch, epochSize)
    file.write(string) 
    file.close()
    


def loadModelTest():
  
   
    json_file = open('Models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights("Models/model.h5") 
    model.compile(optimizer = Adam(lr = learningRate), loss = 'mean_squared_error', metrics = ['accuracy'])
    print("Loaded model from disk")
  

    datagen = ImageDataGenerator(rescale=1./255)
    testGenerator = datagen.flow_from_directory('IndividualData/Deepfake/Testing/', class_mode='binary', batch_size=batchSize, shuffle=False)
   
    results = model.evaluate(testGenerator, steps=testingSize/batchSize)
    
    print('test loss, test acc:', results)
   

def predictModel():
  
    json_file = open('Models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("Models/model.h5") 
    model.compile(optimizer = Adam(lr = learningRate), loss = 'mean_squared_error', metrics = ['accuracy'])
    print("Loaded model from disk")
  

    datagen = ImageDataGenerator(rescale=1./255)
    predictGenerator = datagen.flow_from_directory('IndividualData/Deepfake/Predict/', batch_size=1, shuffle=False)
    data = [None] * predictSize
    predictRaw = [None] * predictSize
    predictData = [None] * predictSize
    i = 0
    for d, l in predictGenerator:
        data[i] = (d)
        predictRaw[i] = model.predict(data[i])
        prediction = model.predict(data[i])
 
        if(prediction > 0.5):
             predictData[i] = "Real"
        else:
             predictData[i] = "Fake"
       
        i += 1
        if i == predictSize:
            break

    
    print(predictData)
    print(predictRaw)
    

def load_images_from_folder(folder):
   cv_img = []
   for img in glob.glob(folder):
       n= cv2.imread(img)
       cv_img.append(n)
   return cv_img


#trainInitial() 
loadModelContinueTraining()
#loadModelTest()  
#predictModel()


    
   
