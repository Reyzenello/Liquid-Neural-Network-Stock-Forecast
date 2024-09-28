# Liquid-Neural-Network-Stock-Forecast

Based on the data of Gold during 2023, it will perform a forecast of the next value:

![image](https://github.com/Reyzenello/Liquid-Neural-Network-Stock-Forecast/assets/43668563/d3f9e9da-4ee3-4131-8929-25c4a4419e3f)


This code implements a "Liquid Neural Network" (LNN) to predict gold prices based on historical data.  The term "Liquid" here doesn't refer to a standard or widely recognized neural network type; it seems to be a custom name for a simple feedforward neural network.

**1. Libraries and Data Preparation:**

* Imports necessary libraries (NumPy, PyTorch, Matplotlib, Scikit-learn).
* Defines historical gold prices for 2023 (you should update this with more current data).
* Converts the data to a NumPy array and scales it using `MinMaxScaler`.
* Creates sequences of length `seq_length` (5 in this case) for training.  For example, the first input sequence would be the first 5 prices, and the corresponding target would be the 6th price.
* Converts the data to PyTorch tensors.
* Creates a `TensorDataset` and `DataLoader` for batching and shuffling during training.

**2. `LiquidLayer` Class:**

```python
class LiquidLayer(nn.Module):
    # ...
```

This class defines a single layer of the LNN.  It's essentially a standard fully connected layer with an activation function.

* `__init__`: Initializes the layer's weights and bias with random values and sets the activation function.
* `forward`: Performs the forward pass, applying a linear transformation and the activation function.

**3. `LiquidNeuralNetwork` Class:**

```python
class LiquidNeuralNetwork(nn.Module):
    # ...
```

This class defines the entire LNN architecture.

* `__init__`:  Creates a list of `LiquidLayer` instances based on the provided `hidden_sizes`.  It also creates an output layer with an identity activation function (no activation).
* `forward`:  Performs the forward pass through all layers sequentially.

**4. Model Training:**

* Sets model parameters (`input_size`, `hidden_sizes`, `output_size`).
* Initializes the LNN model, the Mean Squared Error (MSE) loss function, and the Adam optimizer.
* **Training loop:**
    * Iterates over epochs and batches.
    * `batch_inputs = batch_inputs.view(batch_inputs.size(0), -1)`: Flattens the input sequences into a vector for each sample in the batch before passing to the LNN.
    * `optimizer.zero_grad()`: Clears the gradients.
    * `outputs = lnn(batch_inputs)`: Forward pass.
    * `loss = criterion(outputs, batch_targets)`: Calculates the loss.
    * `loss.backward()`: Backpropagation.
    * `optimizer.step()`: Updates model parameters.
    * Prints the loss every 100 epochs.

**5. Testing and Plotting:**

* Takes the last `seq_length` values from the scaled data as test input.
* Flattens the test input and converts it to a tensor.
* `with torch.no_grad():`: Disables gradient calculation during testing.
* Makes a prediction using the trained model.
* Inverse transforms the predicted value back to the original scale.
* Prints the predicted price.
* Plots the historical prices and the predicted price.
