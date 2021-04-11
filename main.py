import numpy as np
import os
from sklearn.model_selection import train_test_split
from generator import CustomGenerator
from env_variables import TRAINING_DIR, VALIDATION_DIR, BATCH_SIZE, IMG_SIZE, MARK, EPOCHS
from random import shuffle

data_training = []
names_validation = []

print("Data load started")
for group in os.listdir(TRAINING_DIR):
    for img in os.listdir(TRAINING_DIR + "\\" + group):
        if(group == "PNEUMONIA"):
            data_training.append([img, 1])
        elif (group == "NORMAL"):
            data_training.append([img, 0])
shuffle(data_training)
print("Data loaded to training generator")
for img in os.listdir(VALIDATION_DIR):
    names_validation.append(img)
print("Data loaded to validation generator")

names_training = np.array(names_training)
names_validation = np.array(names_validation)

print("Data split")

traning_generator = CustomGenerator(names_training, BATCH_SIZE)
val_generator = CustomGenerator(names_validation, BATCH_SIZE)
