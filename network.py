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
    def reverse_convolute(self, channel, delta): # The delta is related to a kernel deciding the strenght
        channel_x_len = channel.shape[1]
        channel_y_len = channel.shape[0]
        n = delta.shape[0]
        m = delta.shape[1]
        k_d = np.full((1 + channel_y_len - n, 1 + channel_x_len - m), 0.0)
        for x in range(0, 1 + channel_x_len - n):
            for y in range(0, 1 + channel_y_len - m):
                k_d[y][x] = np.multiply(delta, channel[y : y + n, x : x + m]).sum()
        return k_d

    def vector_to_matrix(self, vector):
        size = vector.shape[1]
        matrix = np.full((size, size), 0.0)
        i = 0
        for y in range(size):
            for x in range(size):
                matrix[y][x] = vector[0][i % size]
                i = i + 1
        return matrix

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
                output_activation = activations[-1 - i]
                delta_length = output_activation[0].shape[0] * output_activation[0].shape[1]

                # For hver kernel regn ut en delta tilhørende hver channel.
                for delta_index, kernel in enumerate(layer.get_kernels()):
                    delta = deltas[-1][delta_index * delta_length : (delta_index + 1) * delta_length]
                    delta_n = output_activation[0].shape[0]
                    delta_m = output_activation[0].shape[1]
                    kernel_delta = delta.reshape(delta_n, delta_m)

                    num_channels = len(input_activation)
                    k_d = np.full((2, 2), 0.0)
                    for channel in input_activation:
                        k_d = k_d + self.reverse_convolute(channel, kernel_delta)
                    k_d = k_d / num_channels
                    w_d.append(k_d)


        return w_d, loss

    def get_loss(self, label, activation, transfer):
        output_layer = self._hidden_layers[-1]
        loss = label - activation
        f = output_layer.derivative(transfer.T)
        a = output_layer.derivative(activation.T)
        loss_d = f @ a @ loss.T
        return loss, loss_d


    def train(self, label, data):
        la = 0.01
        w_d, loss = self.backward_pass(label, data)
        for i, layer in enumerate(reversed(self._hidden_layers)):
            if type(layer) == layers.DenseLayer:
                layer._weights = layer._weights + la * w_d[i]
            elif type(layer) == layers.ConvolutionalLayer:
                for k_index in range(len(layer.get_kernels())):
                    layer._kernels[k_index] = layer._kernels[k_index] + 0.1 * la * w_d[i]
                    i = i + 1
        return loss
