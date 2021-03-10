import numpy as np
import math

class Softmax:
    def __call__(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def __call__(self, z):
        e_x = np.exp(z - np.max(z))
        s = e_x / e_x.sum()
        return s

class DerivativeSoftmax:
    def __init__(self):
        self._softmax = Softmax()
        self._sigmoid = Sigmoid()

    def __call__(self, a):
        return self._softmax(a)* (1.0 - self._softmax(a.T))
        
class Sigmoid:
    def __call__(self, activation):
        for i in range(activation.shape[1]):
            activation[0][i] = 1.0 / (1.0 + np.exp(-1.0 * activation[0][i]))
        return activation 

class DerivativeSigmoid:
    def sigmoid(self, activation):
        size = activation.shape[0]
        try:
            for i in range(size):
                activation[i][0] = 1.0 / (1.0 + math.exp(-1.0 * activation[i][0]))
        except:
            print("Derivative sigmoid overflow")
            print(activation)
            print()
            quit()
        return activation
 
    def __call__(self, activation):
        size = activation.shape[0]
        a_d = np.full((size, size), 0.0)

        activation_col_d = self.sigmoid(activation)
        for i in range(size):
            for j in range(size):
                a_d[i][j] = activation_col_d[i]
        return a_d

class Relu:
    def __call__(self, activation):
        return activation * (activation > 0)

class MatrixDerivativeRelu:
    def __call__(self, activation):
        size = activation.shape[0]
        a_d = np.full((size, size), None)
        for i in range(activation.shape[0]):
            for j in range(activation.shape[0]):
                a_d[i][j] = 1.0 * (activation.item(i, 0) >= 0)
        return a_d

class VectorDerivativeRelu:
    def __call__(self, activation):
        return 1.0 * (activation >= 0)


class MSE:
    def __call__(self, label, prediction):
        loss = np.full((label.shape[1], 1), 0.0)
        for i in range(loss.shape[0]):
            loss[i][0] = (label[0][i] - prediction[0][i]) ** 2
        loss = loss / loss.shape[0]
        return loss

class DerivativeMSE:
    def __call__(self, target, output):
        assert(len(output) == len(target))
        output = copy.copy(output)
        for i in range(0, len(target)):
            output[i] = -2 * (output[i] - target[i])
        return output


