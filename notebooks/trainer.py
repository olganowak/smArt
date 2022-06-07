from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import models, layers, optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from PIL import Image

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
        model = VGG16(input_shape = (input_shape), include_top = False, weights = 'imagenet')
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
        prediction_layer = layers.Dense(15, activation='softmax')


        model = models.Sequential([
            resize_layer,
            base_model,
            flatten_layer,
            dense_layer_2,
            dense_layer_4,
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
                    batch_size=16,
                    callbacks=[es])

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        accuracy = self.model.evaluate(X_test, y_test)[1]
        return accuracy

    def predict(self, X_test):
        genres = ['Rococo',
                 'Baroque',
                 'Abstract_Expressionism',
                 'Pop_Art',
                 'Color_Field_Painting',
                 'Romanticism',
                 'Impressionism',
                 'Cubism',
                 'Northern_Renaissance',
                 'Symbolism',
                 'Realism',
                 'Art_Nouveau_Modern',
                 'Naive_Art_Primitivism',
                 'Post_Impressionism']
        new = np.expand_dims(X_test, axis=0)
        array = self.model.predict(new)
        for x in array:
            i = list(x).index(max(x))
        return genres[i]

    def predict_image(self, image_path, size):
        im = Image.open(image_path)
        im = im.resize(size)
        im = np.array(im)
        return self.predict(im)
