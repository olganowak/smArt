import pandas as pd
import numpy as np
import random, os
from PIL import Image


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
                    with Image.open(g + "/" + g[num]) as im:
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
        return df
