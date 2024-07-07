import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class LiquidLayer(nn.Module):
    def __init__(self, input_size, output_size, activation=nn.Tanh()):
        super(LiquidLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.bias = nn.Parameter(torch.randn(output_size))
        self.activation = activation

    def forward(self, x):
        return self.activation(F.linear(x, self.weight, self.bias))

class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.Tanh()):
        super(LiquidNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(LiquidLayer(input_size, hidden_sizes[0], activation))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(LiquidLayer(hidden_sizes[i-1], hidden_sizes[i], activation))
        
        # Output layer
        self.layers.append(LiquidLayer(hidden_sizes[-1], output_size, nn.Identity()))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Historical gold prices for 2023
gold_prices = [
    1936.00, 1917.90, 1915.00, 1919.80, 1910.90, 1906.60, 1894.60, 1884.90,
    1891.80, 1893.70, 1906.80, 1903.90, 1918.00, 1910.60, 1921.10, 1925.40,
    1931.70, 1941.00, 1934.90, 1939.10, 1948.40, 1964.90, 1945.50, 1945.40,
    1962.30, 1965.30, 1958.50, 1965.30, 1970.40, 1973.70, 1968.00, 1954.00,
    1958.40, 1960.60, 1932.30, 1925.00, 1913.40
]

# Convert the list to a numpy array
gold_prices = np.array(gold_prices).reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler()
gold_prices_scaled = scaler.fit_transform(gold_prices)

# Prepare the dataset
seq_length = 5  # Length of the sequence for LSTM
x = []
y = []

for i in range(len(gold_prices_scaled) - seq_length):
    x.append(gold_prices_scaled[i:i + seq_length])
    y.append(gold_prices_scaled[i + seq_length])

x = np.array(x)
y = np.array(y)

# Convert to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Dataset and DataLoader
dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# Define model parameters
input_size = seq_length
hidden_sizes = [50, 30, 10]
output_size = 1

# Initialize the model, loss function, and optimizer
lnn = LiquidNeuralNetwork(input_size, hidden_sizes, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(lnn.parameters(), lr=0.001)

# Training the model
num_epochs = 2000

for epoch in range(num_epochs):
    for batch_inputs, batch_targets in dataloader:
        # Flatten the input for the Liquid Neural Network
        batch_inputs = batch_inputs.view(batch_inputs.size(0), -1)
        
        optimizer.zero_grad()
        outputs = lnn(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete.")

# Testing the model (using the last available sequence)
test_inputs = gold_prices_scaled[-seq_length:]
test_inputs = torch.tensor(test_inputs, dtype=torch.float32).view(1, -1)
with torch.no_grad():
    test_outputs = lnn(test_inputs).numpy()

# Inverse transform the predicted value
predicted_price = scaler.inverse_transform(test_outputs)
print(f"Predicted next gold price: {predicted_price[0][0]:.2f}")

# Plot the results
plt.plot(np.arange(len(gold_prices)), gold_prices, label='Historical Prices')
plt.plot(np.arange(len(gold_prices) - 1, len(gold_prices) + len(predicted_price) - 1), predicted_price, label='Predicted', color='red')
plt.xlabel('Time')
plt.ylabel('Gold Price')
plt.legend()
plt.show()
