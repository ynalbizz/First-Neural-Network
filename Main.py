import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import Functions as Func


# Initialize Training database
images = load_digits().images
label = load_digits().target

# initialize Network dimensions:
input_size = images[0].flatten()  # Num of neurons of input layer(global layer 1)
hidden_size1 = 40  # Num of neurons of hidden layer 1(global layer 2)
hidden_size2 = 20  # Num of neurons of hidden layer 2(global layer 3)
output_size = 10  # Num of neurons of output layer (global layer 4)

# Initialize Weights and Biases
weight_L12 = np.random.randn(hidden_size1, input_size) * 0.01  # Initialize weights between input and hidden layer 1
bias_L2 = np.random.randn(hidden_size1) * 0.01  # hidHen layer 1 bias

weight_L23 = np.random.randn(hidden_size2, hidden_size1) * 0.01  # Initialize weights between hidden layer 1 and 2
bias_L3 = np.random.randn(hidden_size2) * 0.01  # Hidden layer 2 bias

weight_L34 = np.random.randn(output_size, hidden_size2) * 0.01  # Initialize weights between hidden layer 2 and output
bias_L4 = np.random.randn(output_size) * 0.01  # Output bias

# Training settings
learning_rate = 0.1
tolerance = 1e-6
epoch = 0  # just initialize an epoch counting

# Training
while True:
    # Initialize Training vars
    total_cost = 0
    image_id = 0

    # Initialize First Epoch
    for image in images:
        layer_1 = image.flatten()
        # Forward pass

        layer_2 = Func.sigmoid(np.dot(weight_L12, layer_1) + bias_L2)
        layer_3 = Func.sigmoid(np.dot(weight_L23, layer_2) + bias_L3)
        output = Func.sigmoid(np.dot(weight_L34, layer_3) + bias_L4)

        # Cost function
        target = np.zeros(10)
        target[label[image_id]] = 1  # Assuming one-hot encoding for the target
        cost_value = Func.cost(output, target)
        image_id += 1

        # Backpropagation
        output_error = output - target
        output_delta = output_error * Func.sigmoid_derivative(output)

        layer_3_error = np.dot(weight_L34.T, output_delta)
        layer_3_delta = layer_3_error * Func.sigmoid_derivative(layer_3)

        layer_2_error = np.dot(weight_L23.T, layer_3_delta)
        layer_2_delta = layer_2_error * Func.sigmoid_derivative(layer_2)

        # Update weights and biases
        weight_L34 -= learning_rate * np.outer(output_delta, layer_3)
        bias_L4 -= learning_rate * output_delta

        weight_L23 -= learning_rate * np.outer(layer_3_delta, layer_2)
        bias_L3 -= learning_rate * layer_3_delta

        weight_L12 -= learning_rate * np.outer(layer_2_delta, layer_1)
        bias_L2 -= learning_rate * layer_2_delta

        total_cost += cost_value

    avg_cost = total_cost / len(images)
    print(avg_cost)

    if avg_cost < tolerance:
        break

# Test
while True:
    # Starts the test
    number = int(input("Input a number between 0 and 9"))
    number2 = int(input("Input a number between 0 and 178"))
    image_in = 10*number2 + number
    image = images[image_in]
    layer_1 = image.flatten()

    # Forward pass

    layer_2 = Func.sigmoid(np.dot(weight_L12, layer_1) + bias_L2)
    layer_3 = Func.sigmoid(np.dot(weight_L23, layer_2) + bias_L3)
    output = Func.sigmoid(np.dot(weight_L34, layer_3) + bias_L4)

    print("Numero identificado:", np.argmax(output))
    plt.matshow(images[number])
    plt.show()
