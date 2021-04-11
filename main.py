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

data_training = np.array(data_training)
names_validation = np.array(names_validation)

print("Data split")

traning_generator = CustomGenerator(names_training, BATCH_SIZE)
val_generator = CustomGenerator(names_validation, BATCH_SIZE)


def train(epochs, device, learning_rate=1e-3):
    best_model = None
    accuracies = []
    training_losses = []
    validation_losses = []
    cnn = create_squeezenet().to(device)
    summary(cnn, (3, 224, 224))
    ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    max_accuracy = 0

    for epoch in range(epochs):
        losses = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = ce(pred, labels)
            losses.append(loss)
            loss.backward()
            optimizer.step()
        accuracy = float(validate(cnn, validation_loader))
        training_loss = float(sum(losses) / len(losses))
        validation_loss = float(cel(cnn, validation_loader, ce))
        accuracies.append(accuracy)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        if accuracy > max_accuracy:
            best_model = copy.deepcopy(cnn)
            max_accuracy = accuracy
            print("Saving best model with accuracy: ", accuracy)
            torch.save(best_model.state_dict(), MODELPATH)
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
© 2021 GitHub, Inc.