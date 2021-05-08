from env_variables import TRAIN_SOURCE, TRAIN_DEST, VAL_SOURCE, VAL_DEST, TEST_SOURCE, TEST_DEST
import os
from shutil import copyfile

#CREATE FILE SYSTEM
os.makedirs(TRAIN_DEST + "/VIRUS")
os.makedirs(TRAIN_DEST + "/BACTERIA")
os.makedirs(VAL_DEST + "/VIRUS")
os.makedirs(VAL_DEST + "/BACTERIA")
os.makedirs(TEST_DEST + "/VIRUS")
os.makedirs(TEST_DEST + "/BACTERIA")

for path in os.listdir(TRAIN_SOURCE):
    if path.find("virus") != -1:
        copyfile(os.path.join(TRAIN_SOURCE, path), TRAIN_DEST + "/VIRUS/" + path)
    elif path.find("bacteria") != -1:
        copyfile(os.path.join(TRAIN_SOURCE, path), TRAIN_DEST + "/BACTERIA/" + path)

for path in os.listdir(VAL_SOURCE):
    if path.find("virus") != -1:
        copyfile(os.path.join(VAL_SOURCE, path), VAL_DEST + "/VIRUS/" + path)
    elif path.find("bacteria") != -1:
        copyfile(os.path.join(VAL_SOURCE, path), VAL_DEST + "/BACTERIA/" + path)

for path in os.listdir(TEST_SOURCE):
    if path.find("virus") != -1:
        copyfile(os.path.join(TEST_SOURCE, path), TEST_DEST + "/VIRUS/" + path)
    elif path.find("bacteria") != -1:
        copyfile(os.path.join(TEST_SOURCE, path), TEST_DEST + "/BACTERIA/" + path)
