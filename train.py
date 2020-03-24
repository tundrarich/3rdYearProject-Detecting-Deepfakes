# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:59:53 2020

@author: Jay
"""
from PIL import Image
from keras import layers, models
from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
from keras_preprocessing.image import ImageDataGenerator
import modelStructure
import tensorflow as tf
import cv2
import os
import glob
batchSize = 50
epochSize = 5

#trainingSize = 14943
#validationSize = 9467
#testingSize = 8970


trainingSize = 8646
validationSize = 7104
testingSize = 3707

def trainInitial():
    
    model = modelStructure.MesoInception4()
    
    datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=60, zoom_range=0.4, rescale=1./255)
    datagenVal = ImageDataGenerator(rescale=1./255)
    #datagen = ImageDataGenerator(rescale=1./255)
    
    trainGenerator = datagen.flow_from_directory('Meso/Training/', class_mode='binary', batch_size=batchSize, shuffle=False)
    validationGenerator = datagenVal.flow_from_directory('Meso/Validation/', class_mode='binary', batch_size=batchSize, shuffle=False)  
  
    model.fit(trainGenerator, validationGenerator, batchSize, epochSize, trainingSize, validationSize, 0)
    print("")
    model.save()


def loadModelContinueTraining():

    startEpoch = 1
    model = modelStructure.MesoInception4()
    # load weights into new model
    model.loadWeights("Models/model.h5")
    print("Loaded model from disk")
   

    datagen = ImageDataGenerator()
    datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=60, zoom_range=0.4, rescale=1./255)
    datagenVal = ImageDataGenerator(rescale=1./255)
    
    trainGenerator = datagen.flow_from_directory('Meso/Training/', class_mode='binary', batch_size=batchSize, shuffle=False)
    validationGenerator = datagenVal.flow_from_directory('Meso/Validation/', class_mode='binary', batch_size=batchSize, shuffle=False)  
    
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
    print('test loss, test acc:', results)
   



def load_images_from_folder(folder):
   cv_img = []
   for img in glob.glob(folder):
       n= cv2.imread(img)
       cv_img.append(n)
   return cv_img

trainInitial() 
#loadModelContinueTraining()
#loadModelTest()  
