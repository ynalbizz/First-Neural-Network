import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
imageindex = 3
images = load_digits().images
label = load_digits().target

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def forward_pass(weight_L12, bias_L2, weight_L23, bias_L3, weight_L34, bias_L4, layer_1_neurons):
    layer_2 = sigmoid(np.dot(weight_L12, layer_1_neurons) + bias_L2)
    layer_3 = sigmoid(np.dot(weight_L23, layer_2) + bias_L3)
    output = sigmoid(np.dot(weight_L34, layer_3) + bias_L4)
    return layer_2, layer_3, output

def backpropagate(weight_L12, bias_L2, weight_L23, bias_L3, weight_L34, bias_L4, layer_1_neurons, y_true,
                  learning_rate):
    # Forward pass
    layer_2, layer_3, output = forward_pass(weight_L12, bias_L2, weight_L23, bias_L3, weight_L34, bias_L4,
                                            layer_1_neurons)

    # Calculate loss
    loss = mse_loss(y_true, output)

    # Backward pass
    output_error = mse_loss_derivative(y_true, output) * sigmoid_derivative(output)
    layer_3_error = np.dot(weight_L34.T, output_error) * sigmoid_derivative(layer_3)
    layer_2_error = np.dot(weight_L23.T, layer_3_error) * sigmoid_derivative(layer_2)

    # Gradients for weights and biases
    weight_L34_gradient = np.dot(output_error, layer_3.T)
    bias_L4_gradient = output_error

    weight_L23_gradient = np.dot(layer_3_error, layer_2.T)
    bias_L3_gradient = layer_3_error

    weight_L12_gradient = np.dot(layer_2_error, layer_1_neurons.T)
    bias_L2_gradient = layer_2_error

    # Update weights and biases
    weight_L34 -= learning_rate * weight_L34_gradient
    bias_L4 -= learning_rate * bias_L4_gradient

    weight_L23 -= learning_rate * weight_L23_gradient
    bias_L3 -= learning_rate * bias_L3_gradient

    weight_L12 -= learning_rate * weight_L12_gradient
    bias_L2 -= learning_rate * bias_L2_gradient

    return weight_L12, bias_L2, weight_L23, bias_L3, weight_L34, bias_L4, loss

# Inicialização dos pesos e biases
weight_L12 = np.random.randn(40, 64)
bias_L2 = np.random.randn(40, 1)  # Vetor coluna
weight_L23 = np.random.randn(20, 40)
bias_L3 = np.random.randn(20, 1)  # Vetor coluna
weight_L34 = np.random.randn(10, 20)
bias_L4 = np.random.randn(10, 1)  # Vetor coluna

layer_1 = []
for line in images[imageindex]:
    for item in line:
        layer_1.append(item)

layer_1_neurons = np.asarray(layer_1).reshape(-1, 1)  # Vetor coluna

# Dados de saída (meta) - convertendo para um array NumPy
meta = np.zeros(10)
meta[label[imageindex]] = 0
y_true = meta.reshape(-1, 1)  # Vetor coluna

# Parâmetros do treinamento
learning_rate = 0.01
epochs = 1000

# Treinamento
for epoch in range(epochs):
    weight_L12, bias_L2, weight_L23, bias_L3, weight_L34, bias_L4, loss = backpropagate(
        weight_L12, bias_L2, weight_L23, bias_L3, weight_L34, bias_L4, layer_1_neurons, y_true, learning_rate)

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

        print(y_true)
        plt.matshow(images[imageindex])
        plt.show()
print('Training completed.')
