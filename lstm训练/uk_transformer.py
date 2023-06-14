import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set the seed
my_seed = 2023
torch.manual_seed(my_seed)
np.random.seed(my_seed)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
all_data = np.fromfile("uk_data")
stand_scaler = MinMaxScaler()
all_data = stand_scaler.fit_transform(all_data.reshape(-1, 1))

sequence_len = 10
X = []
Y = []
for i in range(len(all_data) - sequence_len):
    X.append(all_data[i:i + sequence_len])
    Y.append(all_data[i + sequence_len])
X = np.array(X)
Y = np.array(Y)

# Split the data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

# Convert the data to PyTorch tensors and move to GPU
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)

# Define the Transformer model using PyTorch
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.input_dim = 1
        self.model_dim = 10  # ensure model_dim is divisible by nhead
        self.nhead = 10
        self.num_layers = 6
        self.fc_in = nn.Linear(self.input_dim, self.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.fc_out = nn.Linear(self.model_dim * sequence_len, 1)

    def forward(self, x):
        x = self.fc_in(x)  # transform input dimension to model dimension
        x = self.transformer_encoder(x.unsqueeze(0))
        x = self.fc_out(x.view(x.size(0), -1))
        return x

# Create the model and move to GPU
transformer = TransformerModel().to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(transformer.parameters(), lr=0.001)

# Training loop
num_epochs = 250
for epoch in range(num_epochs):
    for i in range(len(X_train)):
        optimizer.zero_grad()
        output = transformer(X_train[i])
        loss = criterion(output, Y_train[i].unsqueeze(-1))
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# Make predictions on the test set
transformer.train()

X_test = X_test.view(-1, sequence_len, 1).to(device)
Y_predict = transformer(X_test).squeeze().cpu()
Y_predict_real = stand_scaler.inverse_transform(Y_predict.squeeze().numpy().reshape(-1, 1))
Y_test_real = stand_scaler.inverse_transform(Y_test.cpu().numpy().reshape(-1, 1))

# Save the model parameters
torch.save(transformer.state_dict(), 'uk_transformer_py.pth')

# Plot the results
fig = plt.figure(figsize=(20, 2))
plt.plot(Y_predict_real, label='Predictions')
plt.plot(Y_test_real, label='Actual')
plt.legend()
plt.show()

# Evaluation metrics
def MAPE(true, pred):
    diff = np.abs(np.array(true) - np.array(pred))
    return np.mean(diff / true)

def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

print(f"Root Mean Squared Error (RMSE): {RMSE(Y_predict_real, Y_test_real)}")
print(f"Mean Absolute Percentage Error (MAPE): {MAPE(Y_predict_real, Y_test_real)}")