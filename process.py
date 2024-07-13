"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
CS131   - Artificial Intelligence
A6      - Artificial Neural Network

Implementation of file processing, user interaction, and other helper functions.

by Madelyn Silveira
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import sys
import numpy as np

# Reads input data and populates array of tuples of (input attributes, flower_type)
def read_data():
    with open('iris_data.txt', 'r') as f:
        data = []
        answers = []
        for line in f:
            line = line.strip().split(',')
            inputs = [float(x) for x in line[:-1]]
            name = line[-1]
            data.append(inputs)
            answers.append(name_to_num(name))
    return np.array(data), np.array(answers)

# scalar encoding for correct answers 0-2
def name_to_num(name):
    if name == "Iris-setosa":
        return 0
    elif name == "Iris-versicolor":
        return 1
    elif name == "Iris-virginica":
        return 2
    else:
        return 3 # shouldn't happen

def num_to_name(num):
    if num == 0:
        return "Iris-setosa"
    elif num == 1:
        return "Iris-versicolor"
    elif num == 2:
        return "Iris-virginica"
    else:
        return "Unidentified"

# Prompts the user for the input attributes of an Iris plant
# Outputs the input layer
def prompt():
    sepal_length = float(input('Enter sepal length in cm: '))
    sepal_width = float(input('Enter sepal width in cm: '))
    petal_length = float(input('Enter petal length in cm: '))
    petal_width = float(input('Enter petal width in cm: '))
    return [sepal_length, sepal_width, petal_length, petal_width]


