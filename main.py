import numpy as np
import os
import torch
from generator import DataHandler
from env_variables import TRAINING_DIR, VALIDATION_DIR, TEST_DIR, BATCH_SIZE, IMG_SIZE, MARK, EPOCHS, MODEL_PATH
from random import shuffle
from model import Net
from resources import validate, cel
import matplotlib.pyplot as plt
import copy

generator_training = DataHandler(TRAINING_DIR)
generator_validation = DataHandler(VALIDATION_DIR)
generator_test = DataHandler(TEST_DIR)

set_training = torch.utils.data.DataLoader(generator_training)
set_validation = torch.utils.data.DataLoader(generator_validation)
set_test = torch.utils.data.DataLoader(generator_test)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(torch.cuda.get_device_name(0))
print("Device used: ", device)


def train(epochs, device, learning_rate=1e-3):
    best_model = None
    accuracies = []
    training_losses = []
    validation_losses = []
    cnn = Net().to(device)
    ce = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    max_accuracy = 0
    if device.type == 'cuda':
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    batch_num = len(set_training)
    for epoch in range(epochs):
        losses = []
        for i, (images, labels) in enumerate(set_training):
            labels = labels[0, :]
            if len(labels) != BATCH_SIZE:
                break
            labels = labels.to(device, dtype=torch.long)
            images = images[0, :, :, :]
            images = images.to(device, dtype=torch.float)
            print("Epoch ", epoch + 1, " of ", epochs, ", Batch ", i + 1, " of ", batch_num)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = ce(pred, labels)
            losses.append(loss)
            loss.backward()
            optimizer.step()
        accuracy = float(validate(cnn, set_validation))
        training_loss = float(sum(losses) / len(losses))
        validation_loss = float(cel(cnn, set_validation, ce))
        accuracies.append(accuracy)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("Saving best model with accuracy: ", accuracy)
            torch.save(best_model.state_dict(), MODEL_PATH)
        print("Epoch: ", epoch + 1, ", Accuracy: ", accuracy, "%", ", Trainig error: ", training_loss, ", Validation "
                                                                                                       "error: ",
              validation_loss)
        plt.plot(training_losses, label='training')
        plt.plot(validation_losses, label='validation')
        plt.legend()
        plt.show()
    return best_model


lenet = train(EPOCHS, device)
print("Training finished")
