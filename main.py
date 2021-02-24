import layers
import utility
import network
import numpy as np

def main():
    relu = utility.Relu()
    relu_d = utility.MatrixDerivativeRelu()

    sigmoid = utility.Sigmoid()
    sigmoid_d = utility.DerivativeSigmoid()

    data = np.random.rand(5, 5).flatten().reshape(1, 25)
    label = np.random.rand(1, 3)

    input_layer = layers.InputLayer(25, 5)
    hidden_layer0 = layers.DenseLayer(5, 4, relu, relu_d)
    hidden_layer1 = layers.DenseLayer(4, 3, relu, relu_d)
    output_layer = layers.DenseLayer(3, 3, sigmoid, sigmoid_d)
    hidden_layers = [hidden_layer0, hidden_layer1, output_layer]

    nn = network.Network(input_layer, hidden_layers)
    for i in range(10000):
        print(nn.train(label, data).sum())
        #nn.train(label, data).sum()




        

        

if __name__ == "__main__":
    main()
