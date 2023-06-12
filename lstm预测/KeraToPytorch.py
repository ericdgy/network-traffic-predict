import torch
from torch import nn
import torch.optim as optim


class LSTMModel(nn.Module):
    def __init__(self, activation_name):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=512, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=128, batch_first=True)

        if activation_name == 'relu':
            self.activation = nn.ReLU()
        elif activation_name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_name == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Invalid activation function")

        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])
        out = self.activation(out)
        return out


def build_model(activation_name):
    model = LSTMModel(activation_name)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    return model, optimizer, loss_fn