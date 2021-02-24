import numpy as np

class InputLayer:
    def __init__(self, input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size
        self._weights = np.random.rand(input_size, output_size) 

    def forward(self, data): 
        return data @ self._weights
        
class ConvolutionalLayer:
    def __init__(self, kernel_n, kernel_m, channels, h_stride, v_stride, padding):
        self._kernels = []
        for _ in range(channels):
            self._kernels.append(np.random.rand(kernel_n, kernel_m))
        self._channels = channels
        self._h_stride = h_stride 
        self._v_stride = v_stride 
        self._padding = padding

    def convert_input_shape(self, data):
        if data.ndim == 2:
            print(data.shape)
            data = data.reshape(self._channels, data.shape[0], data.shape[1])

    def forward(self, data, debug=False):
        #data = self.convert_input_shape(data)

        layer_output = []
        for channel in data:
            channel = channel.reshape(1, 10)
            channel_x_len = channel.shape[1]
            channel_y_len = channel.shape[0]
            v = self._v_stride
            h = self._h_stride
            for i, kernel in enumerate(self._kernels):
                n = kernel.shape[0]
                m = kernel.shape[1]
                if debug:
                    print(channel)
                    print()
                    print(kernel)
                    print()
                kernel_output = np.full((channel_y_len - n + 1, channel_x_len - m + 1), 0.0)
                for y in range(0, channel_y_len - n + 1, v):
                    for x in range(0, channel_x_len - m + 1, h):
                        if debug:
                            print(channel[y : y + n, x : x + m])
                        kernel_output[y][x] = kernel.dot(channel[y : y + n, x : x + m].T)
                        if debug:
                            print()
                layer_output.append(kernel_output)
        if debug:
            print()
            print(layer_output)
        print("A")
        return layer_output



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
