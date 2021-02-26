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
                a_d = layer.derivative(input_activation.T)
                w = layer._weights
                delta = a_d.T @ layer._weights @ deltas[-1]
                deltas.append(delta)
            elif type(layer) == layers.ConvolutionalLayer:
                pass

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
                pass
        return loss
