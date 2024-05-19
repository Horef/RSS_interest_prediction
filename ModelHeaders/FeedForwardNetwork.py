import numpy as np
from torch import nn
import torch


# implementation of a simple feed forward neural network with batch normalization
class FFN(nn.Module):
    def __init__(self, vocab_size: int = 300, hidden_size: int = 300, num_classes: int = 1):
        """
        Used to initialize the model
        :param vocab_size: size of the input
        :param hidden_size: size of the hidden layer
        :param num_classes: size of the output
        """
        super(FFN, self).__init__()
        self.name = 'Simple Feed Forward Neural Network'

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.bn2 = nn.BatchNorm1d(num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        if num_classes == 1:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.NLLLoss()

    def forward(self, x):
        """
        Used to define the forward pass of the model
        :param x: input to the model
        :return: output of the model
        """
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)

        if self.num_classes == 1:
            out = self.sigmoid(out)
        else:
            out = self.log_softmax(out)
        return out

    def output_to_labels(self, output):
        """
        Used to convert the output of the model to labels
        :param output: output of the model
        :return: labels
        """
        if self.num_classes == 1:
            labels = output > 0.5
            return np.concatenate(labels.astype(int), axis=None)
        else:
            return torch.argmax(output, dim=1)