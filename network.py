import utility
import numpy as np

class Network:
    def __init__(self, input_layer, hidden_layers):
        self._input_layer = input_layer
        self._hidden_layers = hidden_layers

    def forward_pass(self, data):
        activations = []
        transfers = []
        activations.append(self._input_layer.forward(data))

        for layer in self._hidden_layers:
            activation, transfer = layer.forward(activations[-1])
            activations.append(activation)
            transfers.append(transfer)
        return activations, transfers

    def backward_pass(self, label, data):
        activations, transfers = self.forward_pass(data)
        loss = self.get_loss(label, transfers[-1])
        w_d = [] 
        deltas = []

        f = self._hidden_layers[-1].derivative(transfers[-1].T)
        a = self._hidden_layers[-1].derivative(activations[-1].T)
        deltas.append(f @ a @ loss.T)

        for i, layer in enumerate(reversed(self._hidden_layers)):
            w_d.append(activations[-2 - i].T @ deltas[-1].T)
            a_d = layer.derivative(activations[-2 - i].T)
            w = layer._weights
            delta = a_d.T @ layer._weights @ deltas[-1]
            deltas.append(delta)

        return w_d, loss

    def get_loss(self, label, output):
        return label - output
        return np.full((1, 1), (label - output).sum())


    def train(self, label, data):
        la = 0.01
        w_d, loss = self.backward_pass(label, data)
        for i, layer in enumerate(reversed(self._hidden_layers)):
            layer._weights = layer._weights + la * w_d[i]
        return loss

        

             



