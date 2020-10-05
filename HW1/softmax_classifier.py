import numpy as np
from numpy.core.fromnumeric import prod

def softmax_classifier(W, input, label, lamda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - lamda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N, 1), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    # TODO: Put your code here

    label1d = np.argmax(label, axis=1)
    products = np.matmul(input, W)

    prediction = np.argmax(products, axis=1)
    loss = np.power(prediction - label1d, 2).mean()
    gradient = np.zeros(W.shape)

    for sample in range(input.shape[0]):
      gradient += np.matmul((products[sample] - label[sample]).reshape(-1, 1), input[sample].reshape(-1, 1).T).T
    gradient /= input.shape[0]

    ############################################################################

    return loss, gradient, prediction
