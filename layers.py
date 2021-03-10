import numpy as np
import math

class InputLayer:
    def __init__(self, input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size
        self._weights = np.random.rand(input_size, output_size) 
    
    def convert_input_shape(self, data):
        if type(data) == np.ndarray:
            if not data.ndim == 1:
                data = data.flatten()
        return data
    def get_output_shape(self):
        return (1, self._output_size)

    def forward(self, data): 
        data = self.convert_input_shape(data)
        output = (data @ self._weights).reshape(1, self._output_size)
        return output, []
        
class ConvolutionLayer:
    def __init__(self, input_shape, output_shape, h_stride, v_stride, padding, la):
        self._kernels = []
        self._kernel_n = output_shape[0]
        self._kernel_m = output_shape[1] 
        self._la = la
        self._input_shape = input_shape
        for _ in range(output_shape[2]):
            self._kernels.append(np.random.rand(output_shape[0], output_shape[1]) * 0.1)

        self._h_stride = h_stride 
        self._v_stride = v_stride 
        self._padding = padding

    def get_output_shape(self):
        shape_n = self._input_shape[0] - self._kernel_n + 1
        shape_m = self._input_shape[1] - self._kernel_m + 1
        shape_k = self._input_shape[2] * len(self._kernels)
        return (shape_n, shape_m, shape_k)

    def convert_input_shape(self, data):
        if data[0].ndim == 1:
            data = data.reshape(self._input_shape[2], data.shape[0], data.shape[1])
            return data
        return data
    
    def update_weights(self, weights):
        assert(len(weights) == len(self._kernels))
        for i, d_w in enumerate(weights):
            self._kernels[i] = self._kernels[i] - self._la * d_w

    def get_kernels(self):
        return self._kernels

    def forward(self, data):
        data = self.convert_input_shape(data)

        layer_activations = []
        for channel in data:
            channel_x_len = channel.shape[1]
            channel_y_len = channel.shape[0]
            v = self._v_stride
            h = self._h_stride
            for i, kernel in enumerate(self._kernels):
                n = kernel.shape[0]
                m = kernel.shape[1]
                kernel_activation = np.full((channel_y_len - n + 1, channel_x_len - m + 1), 0.0)
                for y in range(0, channel_y_len - n + 1, v):
                    for x in range(0, channel_x_len - m + 1, h):
                        kernel_activation[y][x] = np.multiply(kernel, channel[y : y + n, x : x + m]).sum()
                layer_activations.append(kernel_activation)
        return layer_activations, []

class DenseLayer:
    def __init__(self, input_shape, output_size, af, af_d, la):
        self._la = la
        self._input_size = math.prod(input_shape)
        self._output_size = output_size
        self._af = af
        self._af_d = af_d
        self._weights = np.random.rand(self._input_size, output_size) * 0.1
    def update_weights(self, d_w):
        self._weights = self._weights + self._la * d_w

    def convert_input_shape(self, data):
        if type(data) is list:
            data = np.squeeze(data).flatten()
            data = data.reshape(1, data.shape[0])
            return data
        else:
            return data

    def get_output_shape(self):
        return (1, self._output_size)

    def update_weights(self, d_w):
        self._weights = self._weights + self._la * d_w

    def forward(self, data):
        data = self.convert_input_shape(data)
        activation = (data @ self._weights)
        transfer = self._af(activation)
        return activation, transfer

    def derivative(self, data):
        return self._af_d(data)
