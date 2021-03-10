import re
import layers
from layers import InputLayer, DenseLayer, ConvolutionLayer
from network import Network
import utility


class GeneratorConfig:
    def __init__(self):
        self._data = {
            "size" : None,
            "noise" : None,
            "training_set_size" : None,
            "validation_set_size" : None,
            "shapes" : None
        }

    def get_config(self):
        self._read_config()
        return self._data

    def _read_config(self):
        with open("./generator_config.txt", "r") as f:
            config_data = f.readlines()

        for line in config_data:
            line = line.strip("\n")
            try:
                variable, value = line.split(": ")
                if variable in self._data:
                    self._parse_data(variable, value)
            except:
                print(variable + " is missing a value.")

    def _parse_data(self, variable, data):
        if variable == "size":
            self._data[variable] = int(data)
        elif variable == "noise":
            self._data[variable] = float(data)
        elif variable == "training_set_size":
            self._data[variable] = int(data)
        elif variable == "validation_set_size":
            self._data[variable] = int(data)
        elif variable == "shapes":
            data = data.replace("]", "").replace("[", "").replace(" ", "").split(",")
            shapes = []
            for value in data:
                shapes.append(data)
            self._data[variable] = data

class NetworkConfig:
    def __init__(self, input_size, output_size):
        self._layers = []
        self._input_size = input_size
        self._output_size = output_size
    
    def get_network(self):
        self._read_config()

        input_layer = None
        layers = []

        prev_layer = None
        for data in self._layers:
            if data["type"] == "input":
                input_size = self._input_size * self._input_size
                output_size = int(data["output_size"])
                layer = InputLayer(input_size, output_size)
            elif data["type"] == "dense":
                if "output_size" in data:
                    output_size = int(data["output_size"])
                else:
                    output_size = self._output_size
                activation_function_str = data["af"]
                activation_function = self._lookup_activation_function(activation_function_str)
                activation_function_d = self._lookup_activation_function_d(activation_function_str)
                learning_rate = float(data["la"])
                layer = DenseLayer(prev_layer.get_output_shape(), output_size, activation_function, activation_function_d, learning_rate)
            elif data["type"] == "convolution":
                if prev_layer == None:
                    input_shape = (self._input_size, self._input_size, 1)
                else:
                    input_shape = prev_layer.get_output_shape()
                kernel_n = int(data["kernel_n"])
                kernel_m = int(data["kernel_m"])
                channels_out = int(data["channels"])
                output_shape = (kernel_n, kernel_m, channels_out)
                v_stride = int(data["stride_n"])
                h_stride = int(data["stride_m"])
                padding = int(data["padding"])
                la = float(data["la"])
                layer = ConvolutionLayer(input_shape, output_shape, h_stride, v_stride, padding, la)
            if input_layer == None:
                input_layer = layer
            else:
                layers.append(layer)
            prev_layer = layer

        network = Network(input_layer, layers)
        return network

    def _lookup_loss_function(self, loss_function):
        if loss_function == "mean_squared_error":
            return utility.MSE()
        elif loss_function == "cross_entropy":
            return utility.CrossEntropy()

    def _lookup_loss_function_d(self, loss_function):
        if loss_function == "mean_squared_error":
            return utility.DerivativeMSE()
        elif loss_function == "cross_entropy":
            return utility.DerivativeCrossEntropy()

    def _lookup_activation_function(self, activation_function):
        if activation_function == "relu":
            return utility.Relu()
        elif activation_function == "softmax":
            return utility.Softmax()
        elif activation_function == "sigmoid":
            return utility.Sigmoid()
 
    def _lookup_activation_function_d(self, activation_function):
        if activation_function == "relu":
            return utility.DerivativeRelu()
        elif activation_function == "softmax":
            return utility.DerivativeSoftmax()
        elif activation_function == "sigmoid":
            return utility.DerivativeSigmoid()
            
     
    def _read_config(self):
        with open("./network_config.txt", "r") as f:
            config_data = f.readlines()
        self._parse_config(config_data)

    def _parse_config(self, data):
        layer_data = []
        for line in data:
            line = line.strip("\n")
            if line == "LAYER":
                layer_data.append([]) 
            else:
                if not len(line) == 0:
                    layer_data[-1].append(line)
        for data in layer_data:
            self._parse_layer(data)

    def _parse_layer(self, data):
        self._layers.append({})
        for line in data:
            try:
                variable, value = line.split(": ")
            except:
                print(variable + "is missing a value.")
            self._layers[-1][variable] = value





