import os
import numpy as np
from skimage.io import imread
import skimage.transform
import torchvision
from env_variables import IMG_SIZE, BATCH_SIZE
from random import shuffle


class DataHandler:
    def __init__(self, data_dir):
        data = []
        for group in os.listdir(data_dir):
            for img in os.listdir(data_dir + "\\" + group):
                if (group == "PNEUMONIA"):
                    data.append([group + "/" + img, 1])
                elif (group == "NORMAL"):
                    data.append([group + "/" + img, 0])
        shuffle(data)
        self.image_filenames = data
        self.batch_size = BATCH_SIZE
        self.data_dir = data_dir

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for image in batch:
            img = imread(os.path.join(self.data_dir, image[0]), as_gray=True) / 255
            batch_x.append(skimage.transform.resize(img, (IMG_SIZE, IMG_SIZE)))
            batch_y.append(image[1])
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        return batch_x, batch_y
