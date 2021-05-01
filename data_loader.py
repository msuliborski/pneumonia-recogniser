import os
from random import shuffle

class NamesLoader:

    def get_dataset(data_dir):
        data = []
        for group in os.listdir(data_dir):
            for img in os.listdir(data_dir + "\\" + group):
                if (group == "PNEUMONIA"):
                    data.append([imread((group + "/" + img), as_gray=True) / 255, 1])
                elif (group == "NORMAL"):
                    data.append([group + "/" + img, 0])
        shuffle(data)
        return data