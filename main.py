"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
CS131   - Artificial Intelligence
A6      - Artificial Neural Network

Implementation of an Artificial Neural Network.

by Madelyn Silveira
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import sys
import numpy as np
from process import read_data, prompt
from network import classify, Layer_Dense, Activation_Softmax, Loss_CategoricalCrossentropy

# data preparation
data, answers = read_data()

# initialize the network 
dense1 = Layer_Dense(4, 6) 
activation1 = Activation_Softmax()
dense2 = Layer_Dense(6,3)
activation2 = Activation_Softmax()
learning_rate = .1

# train the network
for epoch in range(1000):
    # forward propagation
    dense1.forward(data)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    # calculate loss and accuracy
    loss = Loss_CategoricalCrossentropy()
    losses = loss.forward(activation2.output, answers)
    batch_loss = loss.calculate(activation2.output, answers)
    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == answers)
    
    # backward propagation
    loss.backward(activation2.output, answers)
    activation2.backward(loss.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # update weights and biases
    dense1.update(learning_rate)
    dense2.update(learning_rate)
    
    # optional printing
    # if epoch % 10 == 0:
    #     print(f"Pass: {epoch}\tLoss: {batch_loss:.3f}\tAccuracy: {accuracy:.3f}")


# classify user input
input = prompt()
classify(input, dense1, activation1, dense2, activation2)

# optional testing
# test = [[4.6, 3.6, 1.0, 0.2], #Iris-setosa
#         [5.7, 3.0, 4.2, 1.2], # Iris-versicolor
#         [6.4, 3.1, 5.5, 1.8]] # Iris-virginica
# classify(test[0], dense1, activation1, dense2, activation2)

