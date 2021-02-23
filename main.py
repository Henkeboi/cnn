import layers
import utility
import network
import numpy as np

def main():
    relu = utility.Relu()
    vector_relu_d = utility.VectorDerivativeRelu()
    matrix_relu_d = utility.MatrixDerivativeRelu()

    data = np.random.rand(5, 5).flatten().reshape(1, 25)
    label = np.random.rand(1, 1)

    input_layer = layers.InputLayer(25, 5)
    hidden_layer0 = layers.HiddenLayer(5, 4, relu, matrix_relu_d)
    hidden_layer1 = layers.HiddenLayer(4, 3, relu, matrix_relu_d)
    output_layer = layers.HiddenLayer(3, 1, relu, matrix_relu_d)
    hidden_layers = [hidden_layer0, hidden_layer1, output_layer]

    nn = network.Network(input_layer, hidden_layers)
    for i in range(1020):
        print(nn.train(label, data))




        

        

if __name__ == "__main__":
    main()
