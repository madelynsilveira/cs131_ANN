"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
CS131   - Artificial Intelligence
A6      - Artificial Neural Network

Implementation of an Artificial Neural Network classes and functions.

by Madelyn Silveira
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from process import num_to_name

# function to classify and output a single input
def classify(input, dense1, activation1, dense2, activation2):
    dense1.forward(input)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    predictions = np.argmax(activation2.output, axis=1)
    flower = num_to_name(predictions[0])
    print(f"Prediction: {flower}")

# layer object
class Layer_Dense:
    def __init__(self, input_size, output_size):
        self.weights = 0.1 * np.random.randn(input_size, output_size) # shape = inputs by neurons
        self.biases = np.zeros((1, output_size))
        self.dweights = None
        self.dbiases = None
        self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        # Calculate gradients
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
    
    def update(self, learning_rate):
        self.weights += -learning_rate * self.dweights
        self.biases += -learning_rate * self.dbiases

# softmax activation object
class Activation_Softmax:
    def forward(self, inputs):
        # exponential values of a batchwise (subtracting max to prevent overflow)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # keepDims for shape
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        # probability distribution per class per sample
        self.output = probabilities 

    # calculate gradient of loss with respect to inputs
    def backward(self, dvalues):
        # empty array for derivatives
        self.dinputs = np.empty_like(dvalues)

        # outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T) # transpose for proper shape
            self.dinputs[index] = np.dot(matrix, single_dvalues)

# Inherited Loss class
class Loss:
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        data_loss = np.mean(sample_losses) # batch loss
        return data_loss

# Categorical cross-entropy object
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # catch for log of 0 probability
        correct_confidences = y_pred_clipped[range(len(y_pred)), y_true] # passing scalar values
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # calculate gradient of loss given inputs
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        # convert labels to one-hot encoding
        if len(y_true.shape) == 1: # always true, but just in case
            y_true = np.eye(len(dvalues), y_true.max() + 1)[y_true]

        # calculate gradient
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
    

