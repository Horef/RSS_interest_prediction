from torch import nn
import torch


# implementation of a simple feed forward neural network with batch normalization
class FFN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes):
        """
        Used to initialize the model
        :param vocab_size: size of the input
        :param hidden_size: size of the hidden layer
        :param num_classes: size of the output
        """
        super(FFN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.bn2 = nn.BatchNorm1d(num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
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
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.log_softmax(out)
        return out


def to_device(x):
    return x.to(torch.device('mps' if torch.backends.mps.is_available() else 'cpu'))


def train_model(model: FFN, X: [torch.Tensor], y: [torch.Tensor], criterion: nn.NLLLoss = nn.NLLLoss(),
                optimizer: torch.optim.Optimizer = torch.optim.Adam, epochs: int = 10) -> FFN:
    """
    Used to train the model
    :param model: model to train
    :param X: list of tensors of input data
    :param y: list of tensors of target data
    :param criterion: loss function
    :param optimizer: which optimizer to use
    :param epochs: number of epochs to train for
    :return: trained model
    """
    model.train()
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}')

        # To calculate the loss for each epoch
        total_loss = 0

        for i in range(len(X)):
            # Forward pass
            current_x = X[i]
            current_y = y[i]

            current_x = to_device(current_x)
            current_y = to_device(current_y)

            # Forward pass
            outputs = model(current_x)
            loss = criterion(outputs, current_y)
            total_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Loss: {total_loss / len(X):.4f}')

    return model
