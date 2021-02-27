import numpy as np

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

    def forward(self, data): 
        data = self.convert_input_shape(data)
        output = (data @ self._weights).reshape(1, self._output_size)
        return output, []
        
class ConvolutionalLayer:
    def __init__(self, kernel_n, kernel_m, input_channels, output_channels, h_stride, v_stride, padding):
        self._kernels = []
        for _ in range(output_channels):
            self._kernels.append(np.random.rand(kernel_n, kernel_m))
        self._input_channels = input_channels
        self._output_channels = output_channels
        self._h_stride = h_stride 
        self._v_stride = v_stride 
        self._padding = padding

    def convert_input_shape(self, data):
        if data[0].ndim == 1:
            data = data.reshape(self._input_channels, data.shape[0], data.shape[1])
            return data
        return data
    
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
                kernel_transfer = np.full((channel_y_len - n + 1, channel_x_len - m + 1), 0.0)
                for y in range(0, channel_y_len - n + 1, v):
                    for x in range(0, channel_x_len - m + 1, h):
                        kernel_activation[y][x] = np.multiply(kernel, channel[y : y + n, x : x + m]).sum()
                layer_activations.append(kernel_activation)
        return layer_activations, []

class DenseLayer:
    def __init__(self, input_size, output_size, af, af_d):
        self._input_size = input_size
        self._output_size = output_size
        self._af = af
        self._af_d = af_d
        self._weights = np.random.rand(input_size, output_size) * 0.1

    def convert_input_shape(self, data):
        if type(data) is list:
            data_converted = np.full((1, self._input_size), 0.0)
            i = 0
            for channel in data:
                for element in channel.flatten():
                    data_converted[0][i] = element
                    i = i + 1
            return data_converted
        else:
            return data

    def forward(self, data):
        data = self.convert_input_shape(data)
        activation = data @ self._weights
        transfer = self._af(activation)
        return activation, transfer

    def derivative(self, data):
        return self._af_d(data)
