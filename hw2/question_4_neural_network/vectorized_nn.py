# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 14:51:41 2018

@author: Jacob
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 22:15:31 2018

@author: Jacob
"""
import pandas as pd
import numpy as np
import math
from numpy import random
import matplotlib.pyplot as plt

# load data into numpy array with colums PDOT_REC  X_REC  XDOT_REC
numFiles = 5
data = np.zeros((1,3))
for i in range(1,numFiles+1):
    str1 = "C:/dev/classes/abe598_code/hw2/question_4_neural_network/data/sim"
    str2 = str(i)
    str3 = ".csv"
    filePath = str1 + str2 + str3
    dataFrame = pd.read_csv(filePath, encoding="utf-8")
    data = np.vstack((data, dataFrame))
# delete row of zeros at the top
data = data[1:, :]

# input data with each row as a training example
X = data[:, 1:3]
numData = np.shape(X)[0]

# output data
y = (data[:, 0])
y = np.reshape(y, (numData, 1))

def sigmoid(x):
    # return the sigmoid
    return 1 / (1 + np.exp(-x))

# return the first derivative of the sigmoid
def sigmoidPrime(x):
    return np.exp(-x) / ((1+np.exp(-x))**2)

def NN(X, y, numEpochs):   
    costs = np.zeros((1, numEpochs))
    numData = np.shape(X)[0]
    
    # initialize layer parameters
    inputLayerSize = 2 # number of parameters of the system
    hiddenLayerSize = 10 
    outputLayerSize = 1 # number of outputs of the system
    
    W1 = np.random.rand(inputLayerSize, hiddenLayerSize)
    W2 = np.random.rand(hiddenLayerSize, outputLayerSize)
    for epoch in range(0, numEpochs):
            
# =============================================================================
                # forward propagation
# =============================================================================
                # layer 1 - input layer
                # z0 is size (5000 x 2)
                z0 = X
# =============================================================================
                # layer 2 - hidden layer
                # z1 -> (5000x2)
                # W1 -> (2x10)
                # z2 is size (5000 x 10)
                z2 = np.dot(z0, W1)
                # use the activation function
                a2 = sigmoid(z2)
# =============================================================================
                # layer 3 - output layer
                # a2 is size (1x10)
                # W2 is size (10x1)
                
                # size of z3 is (1x1)
                z3 = np.dot(a2, W2)
                
                # yHat is the output of the neural net, a guess for what pDot is given 
                # the inputs phi, p
                # size of yHat is (1x1)
                yHat = sigmoid(z3)
                        
                cost = (1/2) * np.sum(y-yHat)**2 
# =============================================================================
                # backpropagation
# =============================================================================
                learningRate = .1
                
                delta3 = np.multiply(-(y-yHat), sigmoidPrime(z3))
                dJdW2 = np.dot(a2.T, delta3)
                
                delta2 = np.dot(delta3, W2.T)* sigmoidPrime(z2)
                dJdW1 = np.dot(X.T, delta2)
                        
                W1 = W1 - learningRate * dJdW1
                W2 = W2 - learningRate * dJdW2
            costs[0, epoch] = cost
            #print(epoch)
    return costs

numEpochs = 50
costs= NN(X,y,numEpochs)        
        
plt.plot(costs)

        
        
        
        
        
        
        