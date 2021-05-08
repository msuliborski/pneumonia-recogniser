EPOCHS = 30
BATCH_SIZE = 8
TRAINING_DIR = "../vb_data/train"
VALIDATION_DIR = "../vb_data/val"
TEST_DIR = "../vb_data/test"
IMG_SIZE = 224
MARK = "vb_resnet18"
MODEL_PATH = "../models/" + MARK

#DATA PREPARATION
TRAIN_SOURCE = "../chest_xray/train/PNEUMONIA"
TRAIN_DEST = "../vb_data/train"
VAL_SOURCE = "../chest_xray/val/PNEUMONIA"
VAL_DEST = "../vb_data/val"
TEST_SOURCE = "../chest_xray/test/PNEUMONIA"
TEST_DEST = "../vb_data/test"
