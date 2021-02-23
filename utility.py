import numpy as np
import math

class Sigmoid:
    def __call__(self, activation):
        for i in range(activation.shape[1]):
            activation[0][i] = 1.0 / (1.0 + math.exp(-1.0 * activation[0][i]))
        return activation 

class DerivativeSigmoid:
    def __call__(self, activation):
        size = activation.shape[0]
        a_d = np.full((size, size), None)
        for i in range(activation.shape[0]):
            for j in range(activation.shape[0]):
                a_d[i][j] = 1 * (activation.item(i, 0))
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
    def __call__(self, target, output):
        mse = 0
        for i in range(0, len(target)):
            mse = mse + (output[i] - target[i]) ** 2
        return mse

class DerivativeMSE:
    def __call__(self, target, output):
        assert(len(output) == len(target))
        output = copy.copy(output)
        for i in range(0, len(target)):
            output[i] = -2 * (output[i] - target[i])
        return output
