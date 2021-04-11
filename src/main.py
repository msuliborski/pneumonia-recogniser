import numpy as np
import os
from sklearn.model_selection import train_test_split
from src.generator import Custom_Generator
from src.env_variables import TRAINING_DIR, VALIDATION_DIR, BATCH_SIZE, IMG_SIZE, MARK, EPOCHS

names_training = []
names_validation = []

print("Data load started")
for img in os.listdir(TRAINING_DIR):
    names_training.append(img)
print("Data loaded to training generator")
for img in os.listdir(VALIDATION_DIR):
    names_validation.append(img)
print("Data loaded to validation generator")

names_training = np.array(names_training)
names_validation = np.array(names_validation)

print("Data split")

traning_generator = Custom_Generator(names_training, BATCH_SIZE)
val_generator = Custom_Generator(names_validation, BATCH_SIZE)
