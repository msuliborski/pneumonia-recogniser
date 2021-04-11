import os
from random import shuffle

class DataLoader:

    def get_dataset(self, data_dir):
        data = []
        for group in os.listdir(data_dir):
            for img in os.listdir(data_dir + "\\" + group):
                if (group == "PNEUMONIA"):
                    data.append([img, 1])
                elif (group == "NORMAL"):
                    data.append([img, 0])
        shuffle(data)
        return data