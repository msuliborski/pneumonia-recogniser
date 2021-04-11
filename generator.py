import os
import numpy as np
from skimage.io import imread
from skimage import color


class CustomGenerator:
    def __init__(self, image_filenames, batch_size):
        self.image_filenames = image_filenames
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        for img in batch:
            rgb = imread(os.path.join(DATADIR, img))/255
            lab = color.rgb2lab(rgb)
            batch_x.append(lab[:, :, 0]/100)
            batch_y.append(lab[:, :, 1:]/128)
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        batch_x = np.expand_dims(batch_x, axis=3)
        return batch_x, batch_y