import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from autograd import grad
def sigmoid(x):
    a = []
    for i in x:
        a.append(1 / (1 + np.exp(-i)))
    return a

def cost(x,meta):
    c= 0
    b = 0
    for a in x:
        b = b +np.square((a-meta[c]))
        c = c + 1
    return b

imageindex = 0
images = load_digits().images
label = load_digits().target

weight_L12 = np.random.randint(-5, 5, size=(40, 64))
bias_L2 = np.random.randint(-5, 5, size=(40))
weight_L23 = np.random.randint(-5, 5, size=(20, 40))
bias_L3 = np.random.randint(-5, 5, size=(20))
weight_L34 = np.random.randint(-5, 5, size=(10, 20))
bias_L4 = np.random.randint(-5, 5, size=(10))

#initializate training
while imageindex < len(images):

# transform the image state in a vector


    print(len(layer_1_nerons))
#create weights and bias mathemectaly
    result = sigmoid(np.dot(weight_L34,sigmoid(np.dot(weight_L23,sigmoid(np.dot(weight_L12,layer_1_nerons)+bias_L2))+bias_L3))+bias_L4)


    custo = cost(result,meta)




    print(len(result))
    print(grad(custo))
    print(images[imageindex])
    plt.matshow(images[imageindex])
    plt.show()
    imageindex = imageindex + 1







