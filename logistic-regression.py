import numpy as np
import os

#print("Shape of the training data set:", training_data.shape)
# print(training_data)


def train(training_data):
    """
    Train a model on the training_data using Logistic Regression.

    Input
        training_data: a two-dimensional numpy-array with shape = [5000, 39] 

    Output
        fitted_model: one dimensional numpy array that captures the model
    """
    X, Y, alpha, training_iterations = training_data[:,
                                                     1:], training_data[:, 0], 0.1, 12000
    X = np.insert(X, 0, 1, axis=1)
    fitted_model, j = np.ones(X.shape[1]), X.shape[0]

    for n in range(training_iterations):
        z = np.dot(X, fitted_model)
        temp = Y - (1/(1 + np.exp(-z)))
        fitted_model = fitted_model + alpha / j * np.dot(temp, X)

    return fitted_model


def classify(testing_data, fitted_model):
    """
    Classify the rows of testing_data using a fitted_model.
    
    Input
        testing_data: a two-dimensional numpy-array with shape = [n_test_samples, 38]
        fitted_model: the output of your train function.
    
    Output
        class_predictions: a numpy array containing the class predictions for each row
        of testing_data.
    """
    n = len(testing_data)
    temp, class_predictions = np.insert(
        testing_data, 0, 1, axis=1).dot(fitted_model), np.empty(n)

    for x in range(n):
        if (1/(1 + np.exp(-(temp[x])))) < 0.5:
            class_predictions[x] = 0
        else:
            class_predictions[x] = 1

    return class_predictions


def accuracy(y_predictions, y_true):
    """
    Calculate the accuracy.

    Input
        y_predictions: a one-dimensional numpy array of predicted classes (0s and 1s).
        y_true: a one-dimensional numpy array of true classes (0s and 1s).

    Output
        acc: accuracy, a float between 0 and 1 
    """
    t, c = len(y_predictions), 0
    for i in range(t):
        if (y_predictions[i] == y_true[i]):
            c = c + 1
    acc = c / t
    return acc


def example_use():
    here = os.path.dirname(os.path.abspath(__file__))
    training_file = os.path.join(here, "data\\training_data.csv")
    training_data = np.loadtxt(open(training_file, "r"), delimiter=",")

    # Classify training data by training on the same data
    class_predictions = classify(training_data[:, 1:], train(training_data))

    # Check data type(s)
    assert(isinstance(class_predictions, np.ndarray))

    # Check data type of array elements
    assert(np.all(np.logical_or(class_predictions == 0, class_predictions == 1)))

    print("Accuracy when classifying training set: ", 100 * accuracy(
        class_predictions, training_data[:, 0]), "%")


example_use()
