from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random

class Trainer():
    def __init__(self, X_train, y_train):
        """
            X: np.array
            y: pandas Series
        """
        self.X_train = X_train
        self.y_train = y_train

    def load_model(self):
        input_shape = (self.height, self.width, 3)
        model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
        return model

    def set_nontrainable_layers(self, model):
        model.trainable = False
        return model

    def add_layers(self, model):
        '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
        resize_layer = layers.Resizing(self.height, self.width, crop_to_aspect_ratio=False)
        base_model = self.set_nontrainable_layers(model)
        flatten_layer = layers.Flatten()
        dense_layer = layers.Dense(1000, activation='relu')
        dense_layer_2 = layers.Dense(500, activation='relu')
        dense_layer_3 = layers.Dense(500, activation='relu')
        dense_layer_4 = layers.Dense(100, activation='relu')
        prediction_layer = layers.Dense(self.num_genres, activation='softmax')


        model = models.Sequential([
            resize_layer,
            base_model,
            flatten_layer,
            dense_layer,
            dense_layer_2,
            dense_layer_3,
            dense_layer_4,
            prediction_layer
        ])
        # $CHALLENGIFY_END
        return model

    def build_model(self, height, width, num_genres = 4):
        # $CHALLENGIFY_BEGIN
        self.height = height
        self.width = width
        self.num_genres = num_genres
        model = self.load_model()
        model = self.add_layers(model)

        model.compile(loss='categorical_crossentropy',
                    optimizer="adam",
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
                    validation_split=0.3,
                    epochs=50,
                    batch_size=16,
                    callbacks=[es])

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        accuracy = self.model.evaluate(X_test, y_test)[1]
        return accuracy
