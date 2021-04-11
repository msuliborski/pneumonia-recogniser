import numpy as np
import os
from generator import CustomGenerator
from data_loader import DataLoader
from env_variables import TRAINING_DIR, VALIDATION_DIR, TEST_DIR, BATCH_SIZE, IMG_SIZE, MARK, EPOCHS
from random import shuffle

data_training = DataLoader.get_dataset(TRAINING_DIR)
data_validation = DataLoader.get_dataset(VALIDATION_DIR)
data_test = DataLoader.get_dataset(TEST_DIR)


traning_generator = CustomGenerator(names_training, BATCH_SIZE)
val_generator = CustomGenerator(names_validation, BATCH_SIZE)
