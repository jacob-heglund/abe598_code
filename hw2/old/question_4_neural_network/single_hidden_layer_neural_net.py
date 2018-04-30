# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:20:01 2018

@author: Jacob
"""
import pandas as pd
import numpy as np
import math
from numpy import random

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

np.random.seed(1)
import numpy as np

class Neural_Network(object):
    def __init__(self):
        # definte hyperparameters
        self.inputLayerSize = 2
        self.hiddenLayerSize = 10
        self.outputLayerSize = 1
        
        # weights (parameters to optimize over)
        self.W1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize)
        
        self.W2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize)
    
    # propagates input through the network
    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
    
    def costFunction(self, X, y):
        # computer cost for given X, y using weights stored in the class
        self.yHat = self.forward(X)
        J = (1/2) * sum((y-self.yHat)**2)
        return J
        
    # compute derivatives with respect to W1 and W2
    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
    
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)
        
        return dJdW2, dJdW1
    
    def backProp(self, X, y):
        self.learningRate = .2
        dJdW2, dJdW1 = self.costFunctionPrime(X, y)
        # do gradient descent on the weights to reduce the cost
        self.W1 = self.W1 - self.learningRate*dJdW1
        self.W2 = self.W2 - self.learningRate*dJdW2

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        # derivative of the sigmoid function
        return np.exp(-z) / ((1+np.exp(-z))**2)
        

NN = Neural_Network()
numEpochs = 10000
    
for i in range(0, numEpochs):
    for j in range(0, numData):
        xData = X[j,:]
        yData = y[j]
    
        NN.backProp(xData,yData)

























