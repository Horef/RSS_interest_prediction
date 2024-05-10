from torch import nn
import torch

# implementation of a simple feed forward neural network with batch normalization
class FFN(nn):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Used to initialize the model
        :param input_size: size of the input
        :param hidden_size: size of the hidden layer
        :param output_size: size of the output
        """
        super(FFN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Used to define the forward pass of the model
        :param x: input to the model
        :return: output of the model
        """
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.log_softmax(out)
        return out