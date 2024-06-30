import numpy as np
import matplotlib.pyplot as plt
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x,-500,500)))

def sigmoid_derivative(x):
    return x * (1 - x)

# Cost function (mean squared error for simplicity)
def cost(output, target):
    return np.mean((output - target) ** 2)

