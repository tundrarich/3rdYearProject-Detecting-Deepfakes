# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:22:03 2020

@author: Jay
"""

from keras import layers, models
from keras.models import Model as KerasModel
from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from keras.optimizers import Adam
from numpy.random import seed
seed(1)


class MesoInception4():
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])
    
    def InceptionLayer(self, a, b, c, d):
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)
            
            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)
            
            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate = 2, strides = 1, padding='same', activation='relu')(x3)
            
            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate = 3, strides = 1, padding='same', activation='relu')(x4)

            y = Concatenate(axis = -1)([x1, x2, x3, x4])
            
            return y
        return func
    
    def init_model(self):
        x = Input(shape = (256, 256, 3))
        
        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)        
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return KerasModel(inputs = x, outputs = y)
    
   # def __init__():
    #    self.model = 0

    
    
    def predictModel(self, x, batchSize):
       return self.model.predict_classes(x)
    
   
    def fit(self, x, y, batchSize, epochSize, trainingSize, validationSize, initialEpoch):
        
       #classWeight = {0:100, 1:50}
       # return self.model.fit_generator(x, steps_per_epoch=256, validation_data=y, validation_steps=64, Epochs=1)
       return self.model.fit_generator(x, steps_per_epoch=trainingSize/batchSize, epochs=epochSize, validation_data=y, initial_epoch=initialEpoch)
    

    def loadWeights(self, path):
        self.model.load_weights(path)    
    
        
    def evaluateModel(self, x, batchSize, testingSize):
       return self.model.evaluate(x, steps=testingSize/batchSize)
    
    def save(self):
        model_json = self.model.to_json()
        with open("Models/model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("Models/model.h5")
        print("Saved model to disk")
        