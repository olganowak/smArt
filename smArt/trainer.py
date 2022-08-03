from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
from PIL import Image
import requests

class Trainer():
    def __init__(self, X_train, y_train, X_val, y_val):
        """
            X: np.array
            y: pandas Series
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def load_model(self):
        input_shape = (self.height, self.width, 3)
        model = ResNet50(input_shape = input_shape, include_top = False, weights = 'imagenet')
        return model

    def set_nontrainable_layers(self, model):
        model.trainable = False
        return model

    def add_layers(self, model):
        '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
        resize_layer = layers.Resizing(self.height, self.width, crop_to_aspect_ratio=False)
        base_model = self.set_nontrainable_layers(model)
        flatten_layer = layers.Flatten()
        dropout_layer = layers.Dropout(0.3)
        dense_layer = layers.Dense(1000, activation='relu')
        dense_layer_2 = layers.Dense(500, activation='relu')
        dense_layer_3 = layers.Dense(500, activation='relu')
        dense_layer_4 = layers.Dense(100, activation='relu')
        prediction_layer = layers.Dense(15, activation='softmax')


        model = models.Sequential([
            resize_layer,
            base_model,
            flatten_layer,
            dense_layer,
            dropout_layer,
            dense_layer_2,
            dropout_layer,
            dense_layer_3,
            dropout_layer,
            dense_layer_4,
            dropout_layer,
            prediction_layer
        ])
        # $CHALLENGIFY_END
        return model

    def build_model(self, height, width):
        # $CHALLENGIFY_BEGIN
        self.height = height
        self.width = width
        model = self.load_model()
        model = self.add_layers(model)
        opt = optimizers.Adam(learning_rate=1e-4)
        #opt = optimizers.SGD(learning_rate=0.0001)
        model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
        return model

    def run(self):
        """set and train the pipeline"""
        self.model = self.build_model(self.height, self.width)
        
        es = EarlyStopping(monitor = 'val_accuracy', 
                   mode = 'max', 
                   patience = 5,
                   verbose = 1, 
                   restore_best_weights = True)
                   
        self.model.fit(self.X_train, self.y_train, 
                    validation_data=(self.X_val, self.y_val), 
                    epochs=50, 
                    batch_size=32, 
                    callbacks=[es])

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the accuracy"""
        accuracy = self.model.evaluate(X_test, y_test)[1]
        return accuracy

    def predict(self, X_test):
        genres = ['Expressionism',
                'Rococo',
                'Baroque',
                'Abstract Expressionism',
                'Pop Art',
                'Color Field Painting',
                'Romanticism',
                'Impressionism',
                'Cubism',
                'Northern Renaissance',
                'Symbolism',
                'Realism',
                'Art Nouveau Modern',
                'Naive Art Primitivism',
                'Post Impressionism']

        new = np.expand_dims(X_test, axis=0)
        #print(new)
        array = self.model.predict(new)
        #print(np.around(array, 2))
        for x in array:
            i = list(x).index(max(x))
        #print(i)
        return genres[i]

    def predict_image(self, image_path, size):
        im = Image.open(requests.get(image_path, stream=True).raw)
        #im = Image.open(image_path)
        im = im.resize(size)
        im = np.array(im)
        im = preprocess_input(
            im, data_format=None
        )
        return self.predict(im)
