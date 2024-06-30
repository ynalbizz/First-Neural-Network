import numpy as np
import matplotlib.pyplot as plt
import Functions as Func
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import sys

# Initialize Training database
(images, labels), (test_img, test_labels) = mnist.load_data()
# initialize Network dimensions:
input_size = len(images[0].flatten())  # Num of neurons of input layer(global layer 1)
hidden_size1 = 256  # Num of neurons of hidden layer 1(global layer 2)
hidden_size2 = 128  # Num of neurons of hidden layer 2(global layer 3)
output_size = 10  # Num of neurons of output layer (global layer 4)

# Initialize Weights and Biases
weight_L12 = np.random.randn(hidden_size1, input_size) * 0.01  # Initialize weights between input and hidden layer 1
bias_L2 = np.random.randn(hidden_size1) * 0.01  # hidHen layer 1 bias

weight_L23 = np.random.randn(hidden_size2, hidden_size1) * 0.01  # Initialize weights between hidden layer 1 and 2
bias_L3 = np.random.randn(hidden_size2) * 0.01  # Hidden layer 2 bias

weight_L34 = np.random.randn(output_size, hidden_size2) * 0.01  # Initialize weights between hidden layer 2 and output
bias_L4 = np.random.randn(output_size) * 0.01  # Output bias

# Training settings
learning_rate = 0.01
hit_tolerance = 80  # hit percentage tolerance in percentage
batch_size = 16
# Just Initialize some Vars
epoch = 0
hit_percentage = 0
avg_cost = 0
# Training
while hit_percentage <= hit_tolerance:
    # Initialize Training vars
    total_cost = 0
    image_id = 0
    hits = 0
    shuffled_indexes = np.random.permutation(np.arange(len(images)))

    # Initialize mini batches
    for i in range(0, len(images), batch_size):
        batch_images = images[shuffled_indexes[i:i+batch_size]]
        batch_labels = labels[shuffled_indexes[i:i + batch_size]]
        batch_progress = 0

        # Train Batch per Batch
        for image, label in zip(batch_images, batch_labels):
            layer_1 = image.flatten()

            sys.stdout.write(f"\r Epochs: {epoch} || Hit Percentage: {hit_percentage:.2f} || Avg. Cost: {avg_cost:.4f} || Batch_{i/batch_size:.0f} Progress: {(batch_progress/len(batch_images))*100:.2f}%")
            sys.stdout.flush()
            # Forward pass
            layer_2 = Func.sigmoid(np.dot(weight_L12, layer_1) + bias_L2)
            layer_3 = Func.sigmoid(np.dot(weight_L23, layer_2) + bias_L3)
            output = Func.sigmoid(np.dot(weight_L34, layer_3) + bias_L4)

            # Cost function
            target = np.zeros(10)
            target[label] = 1  # Assuming one-hot encoding for the target
            cost_value = Func.cost(output, target)

            # Accuracy calculation
            if np.argmax(output) == label:
                hits += 1

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
            batch_progress += 1

    hit_percentage = hits / len(images) * 100
    epoch += 1
    avg_cost = total_cost / len(images)

# Test
while True:
    # Starts the test
    input("Start Test? : ")
    number2 = np.random.randint(0, len(test_img))
    image_in = number2
    image = test_img[image_in]
    layer_1 = image.flatten()

    # Forward pass

    layer_2 = Func.sigmoid(np.dot(weight_L12, layer_1) + bias_L2)
    layer_3 = Func.sigmoid(np.dot(weight_L23, layer_2) + bias_L3)
    output = Func.sigmoid(np.dot(weight_L34, layer_3) + bias_L4)

    print("Numero identificado:", np.argmax(output))

    plt.matshow(images[image_in])
    plt.show()
