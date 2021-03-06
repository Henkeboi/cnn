import layers
import utility
import numpy as np

class Network:
    def __init__(self, input_layer, hidden_layers):
        self._input_layer = input_layer
        self._hidden_layers = hidden_layers

    def forward_pass(self, data):
        activations = []
        transfers = []
        activations.append(self._input_layer.forward(data)[0])

        for layer in self._hidden_layers:
            activation, transfer = layer.forward(activations[-1])
            activations.append(activation)
            transfers.append(transfer)
        return activations, transfers
    
    def vectorize_activation(self, activation):
        i = 0
        length = len(activation) * activation[0].shape[0] * activation[0].shape[1]
        activation_reshaped = np.full((length, 1), 0.0)
        for channel in activation:
            for y in range(channel.shape[0]):
                for x in range(channel.shape[1]):
                    activation_reshaped[i][0] = channel[y][x]
                    i = i + 1
        return activation_reshaped

    # todo: Support stride.
    def convolute(self, channel, delta): # The delta is related to a kernel deciding the strenght
        channel_x_len = channel.shape[1]
        channel_y_len = channel.shape[0]
        n = delta.shape[0]
        m = delta.shape[1]
        k_d = np.full((1 + channel_y_len - n, 1 + channel_x_len - m), 0.0)
        for x in range(0, 1 + channel_x_len - n):
            for y in range(0, 1 + channel_y_len - m):
                k_d[y][x] = np.multiply(delta, channel[y : y + n, x : x + m]).sum()
        return k_d

    # todo: Support stride
    def reverse_convolute(self, delta, kernel):
        stride = 1
        n = kernel.shape[0]
        m = kernel.shape[1]
        channel_x_len = delta.shape[1] + kernel.shape[1] - 1
        channel_y_len = delta.shape[0] + kernel.shape[0] - 1
        channel = np.full((channel_x_len, channel_y_len), 0.0)
        for y in range(0, channel_y_len - n + 1):
            for x in range(0, channel_x_len - m + 1):
                channel[y : y + n, x : x + m] = np.true_divide(delta[y][x], kernel)
        return channel

    def vector_to_matrix(self, vector):
        size = vector.shape[1]
        matrix = np.full((size, size), 0.0)
        i = 0
        for y in range(size):
            for x in range(size):
                matrix[y][x] = vector[0][i % size]
                i = i + 1
        return matrix

    def format_delta(self, layer, next_layer, output_sample, delta):
        if type(delta == np.ndarray):
            if type(next_layer) == layers.DenseLayer:
                delta = delta[-1]
                delta_length = output_sample.shape[0] * output_sample.shape[1]
                delta_n = output_sample.shape[0]
                delta_m = output_sample.shape[1]
                kernel_deltas = []
                for delta_index, kernel in enumerate(layer.get_kernels()):
                    kernel_delta = delta[delta_index * delta_length : (delta_index + 1) * delta_length]
                    kernel_delta = kernel_delta.reshape(delta_n, delta_m)
                    kernel_deltas.append(kernel_delta)
                return kernel_deltas
            elif type(next_layer) == layers.ConvolutionalLayer:
                kernel_deltas = []
                for i in range(len(next_layer.get_kernels())):
                    kernel_deltas.insert(0, delta[-1 - i])
                return kernel_deltas     

    def backward_pass(self, label, data):
        activations, transfers = self.forward_pass(data)
        w_d = [] 
        deltas = []

        loss, loss_d = self.get_loss(label, activations[-1], transfers[-1])
        deltas.append(loss_d)
    
        for i, layer in enumerate(reversed(self._hidden_layers)):
            if type(layer) == layers.DenseLayer:
                input_activation = activations[-2 - i]
                if type(input_activation) == list:
                    input_activation = self.vectorize_activation(input_activation).T
                    w_d.append(input_activation.T @ deltas[-1].T)
                    a_d = self.vector_to_matrix(input_activation).T
                else:
                    w_d.append(input_activation.T @ deltas[-1].T)
                    a_d = layer.derivative(input_activation.T)
                w = layer._weights
                delta = a_d @ w @ deltas[-1]
                deltas.append(delta)
            elif type(layer) == layers.ConvolutionalLayer:
                input_activation = activations[-2 - i]
                output_sample = activations[-1 - i][0]
                assert(not i == 0)
                next_layer = self._hidden_layers[-i] 
                kernel_deltas = self.format_delta(layer, next_layer, output_sample, deltas)

                for delta_index, kernel in enumerate(layer.get_kernels()):
                    kernel_delta = kernel_deltas[delta_index]

                    delta = np.full((input_activation[0].shape[0], input_activation[0].shape[1]), 0.0)
                    k_d = np.full((kernel.shape[0], kernel.shape[0]), 0.0)
                    for channel in input_activation:
                        self.reverse_convolute(kernel_delta, kernel)
                        #delta = delta + channel @ self.reverse_convolute(kernel_delta, kernel)
                        delta = delta + self.reverse_convolute(kernel_delta, kernel)
                        k_d = k_d + self.convolute(channel, kernel_delta)

                    num_channels = len(input_activation)
                    num_channels = 1
                    delta = np.true_divide(delta, num_channels)
                    k_d = np.true_divide(k_d, num_channels)
                    deltas.append(delta)
                    w_d.append(k_d)

        return w_d, loss

    def get_loss(self, label, activation, transfer):
        loss = np.full((label.shape[1], 1), 0.0)
        for i in range(loss.shape[0]):
            loss[i][0] = (label[0][i] - transfer[0][i])


        output_layer = self._hidden_layers[-1]
        f = output_layer.derivative(transfer.T)
        a = activation.T
        #print(loss.shape)
        #print(f.shape)
        #print(a.shape)
        
        loss_d = f.T @ activation.T 
        loss_d = np.multiply(loss, loss_d)
        return loss, loss_d

    def train(self, label, data):
        la_dense = 0.01
        la_convolution = 0.00001
        w_d, loss = self.backward_pass(label, data)
        for i, layer in enumerate(reversed(self._hidden_layers)):
            if type(layer) == layers.DenseLayer:
                layer._weights = layer._weights + la_dense * w_d[i]
            elif type(layer) == layers.ConvolutionalLayer:
                for k_index in range(len(layer.get_kernels())):
                    layer._kernels[k_index] = layer._kernels[k_index] +  la_convolution * w_d[i]
                    i = i + 1
        return loss
