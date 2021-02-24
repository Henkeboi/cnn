import numpy as np

class InputLayer:
    def __init__(self,input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size
        self._weights = np.random.rand(input_size, output_size) 

    def forward(self, data): 
        return data @ self._weights
        

class DenseLayer:
    def __init__(self, input_size, output_size, af, af_d):
        self._input_size = input_size
        self._output_size = output_size
        self._af = af
        self._af_d = af_d
        self._weights = np.random.rand(input_size, output_size) 

    def forward(self, data):
        activation = data @ self._weights
        transfer = self._af(activation)
        return activation, transfer

    def derivative(self, data):
        return self._af_d(data)
