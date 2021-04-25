import torch
from torch import nn
import torchvision
import torchvision.models as models
from env_variables import BATCH_SIZE

def create_model():
    return nn.Sequential(
        nn.Conv2d(1, 4, 5, padding=2),
        nn.ReLU(),
        # nn.AvgPool2d(2, stride=2),
        #
        # nn.Conv2d(4, 16, 5, padding=2),
        # nn.ReLU(),
        # nn.AvgPool2d(2, stride=2),
        #
        # nn.Conv2d(16, 64, 5, padding=2),
        # nn.ReLU(),
        # nn.AvgPool2d(2, stride=2),
        nn.Flatten(),
        nn.Linear(200704, 1000),
        nn.ReLU(),
        nn.Linear(1000, 50),
        nn.ReLU(),
        nn.Linear(50, 2)

        # nn.Flatten(),
        # nn.Linear(50176, 1000),
        # nn.ReLU(),
        # nn.Linear(1000, 50),
        # nn.ReLU(),
        # nn.Linear(50, 2)
    )
