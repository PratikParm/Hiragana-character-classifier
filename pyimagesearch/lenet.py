import logging
from datetime import datetime
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten


class LeNet(Module):
    def __init__(self, num_channels, classes):
        super(LeNet, self).__init__()

        # 1st Conv => ReLU  => Pool
        self.conv1 = Conv2d(in_channels=num_channels,
                            out_channels=20,
                            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2),
                                  stride=(2, 2))

        # 2nd Conv => ReLU  => Pool
        self.conv2 = Conv2d(in_channels=20,
                            out_channels=50,
                            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2),
                                  stride=(2, 2))

        # FC => ReLU
        self.fc1 = Linear(in_features=800, out_features=500)
        self.relu3 = ReLU()

        # Softmax classifier
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logsoftmax = LogSoftmax(dim=1)

    def forward(self, x):

        # Pass the input to the first CONV => RELU => POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # Pass the output from the previous layer to the 2nd set of
        # CONV => RELU => POOL layer
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # Flatten the output from previous layer and pass it through
        # FC => RELU layer
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        # Pass the output to softmax classifier
        x = self.fc2(x)
        output = self.logsoftmax(x)

        return output
