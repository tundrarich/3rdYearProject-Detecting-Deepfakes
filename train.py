# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:59:53 2020

@author: Jay
"""
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
import os
import glob
import numpy as np
from keras.models import load_model
from keras.models import Sequential,model_from_json
batchSize = 50
epochSize = 1
#trainingSize = 14943
#validationSize = 9467
#testingSize = 8970
	


trainingSize = 8646
validationSize = 887
testingSize = 2959

def trainInitial():
    
    model = modelStructure.MesoInception4()
    
    datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=60, zoom_range=0.2, rescale=1./255)
    #datagenVal = ImageDataGenerator()
    #datagen = ImageDataGenerator(rescale=1./255)
    
    trainGenerator = datagen.flow_from_directory('Meso/Training/', class_mode='binary', batch_size=batchSize, shuffle=True)
    validationGenerator = datagen.flow_from_directory('Meso/Validation/', class_mode='binary', batch_size=batchSize, shuffle=True)  
  
    model.fit(trainGenerator, validationGenerator, batchSize, epochSize, trainingSize, validationSize, 0)
    print("")
    model.save()


def loadModelContinueTraining():

    startEpoch = 21
    model = modelStructure.MesoInception4()
    # load weights into new model
    model.loadWeights("Models/model.h5")
    print("Loaded model from disk")
   
    datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=60, zoom_range=0.2, rescale=1./255)
    #datagenVal = ImageDataGenerator()
    
    trainGenerator = datagen.flow_from_directory('Meso/Training/', class_mode='binary', batch_size=batchSize, shuffle=True)
    validationGenerator = datagen.flow_from_directory('Meso/Validation/', class_mode='binary', batch_size=batchSize, shuffle=True)  
    
    model.fit(trainGenerator, validationGenerator, batchSize, epochSize, trainingSize, validationSize, startEpoch - 1)
    print("")
    model.save()
  


def loadModelTest():
  
    model = modelStructure.MesoInception4()
    # load weights into new model
    model.loadWeights("Models/model.h5")

    print("Loaded model from disk")
  

    datagen = ImageDataGenerator(rescale=1./255)
    testGenerator = datagen.flow_from_directory('Meso/Testing/', class_mode='binary', batch_size=batchSize, shuffle=False)
   
    results = model.evaluateModel(testGenerator, batchSize, testingSize)
    #results = model.evaluateModel(testGenerator, steps=testingSize/batchSize)
    print('test loss, test acc:', results)
   

def predictModel():
  
    model = modelStructure.MesoInception4()
    # load weights into new model
    model.loadWeights("Models/model.h5")

    print("Loaded model from disk")
  

    datagen = ImageDataGenerator(rescale=1./255)
    predictGenerator = datagen.flow_from_directory('Meso/Predict/', batch_size=1, shuffle=False)
    data = []
    i = 0
    for d, l in predictGenerator:

        data.append(d)
        #labels.append(l)
        i += 1
        if i == 3:
            break

    plt.figure()
    plt.imshow(data)
    plt.colorbar()
    plt.grid(False)
    plt.show()
   # results = model.predictModel(predictGenerator, 1)
    #results = model.evaluateModel(testGenerator, steps=testingSize/batchSize)
   # print(results)
    

def load_images_from_folder(folder):
   cv_img = []
   for img in glob.glob(folder):
       n= cv2.imread(img)
       cv_img.append(n)
   return cv_img

#trainInitial() 
#loadModelContinueTraining()
#loadModelTest()  
predictModel()
