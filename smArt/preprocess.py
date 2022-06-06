import pandas as pd
import numpy as np
import random, os
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


class Preproc():
    def __init__(self, size, sample_size, file_path, threshold=1450):
        self.threshold = threshold
        self.file_path = file_path
        self.size = size
        self.sample_size = sample_size

    def get_data(self):

        #create the listdir
        main = os.listdir(self.file_path)
        #create the list with all genres
        genres = []
        for genre in main:
            if genre[0] != "." and len(os.listdir(self.file_path + genre)) > self.threshold:
                genres.append(genre)
        print(f"We are going to process {len(genres)} genres for a total of {self.sample_size*len(genres)}")

        paintings_list = []
        counter = 0
        genre_counter = 0
        for genre in genres:
            g = os.listdir(self.file_path + genre)
            g = [x for x in g if x[0] != "."]
            genre_counter += 1
            print(f"We are processing the genre number {genre_counter} which has {len(g)} images")
            if len(g) > self.sample_size:
                i = random.sample(range(len(g)), self.sample_size)
                for num in i:
                    with Image.open(self.file_path + genre + "/" + g[num]) as im:
                        img_resized = im.resize(self.size)
                        image_array = np.array(img_resized)
                        string = g[num][:-4]
                        string = string.replace("-"," ")
                        string = string.split("_", maxsplit=1)
                        string.insert(0, genre)
                        string.append(genre + "/" + g[num])
                        string.append(image_array)
                        paintings_list.append(string)

                        counter += 1
                        if counter % 100 == 0:
                            print(counter)
            else:
                for num in range(len(g)):
                    with Image.open(self.file_path + genre + "/" + g[num]) as im:
                        img_resized = im.resize(self.size)
                        image_array = np.array(img_resized)
                        string = g[num][:-4]
                        string = string.replace("-"," ")
                        string = string.split("_", maxsplit=1)
                        string.insert(0, genre)
                        string.append(genre + "/" + g[num])
                        string.append(image_array)
                        paintings_list.append(string)

                        counter += 1
                        if counter % 100 == 0:
                            print(counter)

                i = random.sample(range(len(g)), self.sample_size - len(g))
                for num in i:
                    with Image.open(g + "/" + g[num]) as im:
                        img_flipped = im.transpose(Image.FLIP_TOP_BOTTOM)
                        img_resized = img_flipped.resize(self.size)
                        image_array = np.array(img_resized)
                        string = g[num][:-4]
                        string = string.replace("-"," ")
                        string = string.split("_", maxsplit=1)
                        string.insert(0, genre)
                        string.append(genre + "/" + g[num])
                        string.append(image_array)
                        paintings_list.append(string)

                        counter += 1
                        if counter % 100 == 0:
                            print(counter)

        df = pd.DataFrame(paintings_list)
        df = df.rename(columns = {0:'genre', 1:'artist', 2:'title', 3:'path', 4:'image'})
        self.df = df
        return df

    def train_test_split(self):
        genres = list(self.df["genre"].unique())
        y = self.df["genre"].map(lambda x: genres.index(x))
        y = to_categorical(y)
        X = np.stack(self.df['image'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        return X_train, X_test, y_train, y_test
