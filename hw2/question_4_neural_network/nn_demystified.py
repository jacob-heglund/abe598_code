# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 21:36:12 2018

@author: Jacob
"""
# created as an exercise copying the work of the series by
# Welch labs on Youtube https://www.youtube.com/watch?v=UJwK6jAStmg

import numpy as np

class Neural_Network(object):
    def __init__(self):
        # definte hyperparameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 10
        
        # weights (parameters to optimize over)
        self.W1 = np.random.rand(self.inputLayerSize, \
                                 self.hiddenLayerSize)
        
        self.W2 = np.random.rand(self.hiddenLayerSize, \
                                 self.outputLayerSize)
    
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
        dJdW2, dJdW1 = self.costFunctionPrime(self, X, y)
        # do gradient descent on the weights to reduce the cost
        self.W1 = self.W1 - self.learningRate*dJdW1
        self.W2 = self.W2 - self.learningRate*dJdW2

    def sigmoid(z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(z):
        # derivative of the sigmoid function
        return np.exp(-z) / ((1+np.exp(-z))**2)
    
    
    
    
    
    
    
    
    
    