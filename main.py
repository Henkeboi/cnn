import layers
import utility
import network
import numpy as np
from generator import Generator

def main():
    relu = utility.Relu()
    relu_d = utility.MatrixDerivativeRelu()

    sigmoid = utility.Sigmoid()
    sigmoid_d = utility.DerivativeSigmoid()

    softmax = utility.Softmax()
    softmax_d = utility.DerivativeSoftmax()

    size = 8
    noise = 0.0
    generator = Generator(size, noise)
    data = generator.generate(500)
    
    input_shape = (size, size, 1)
    output_shape = (3, 3, 1)
    input_layer = layers.ConvolutionalLayer(input_shape, output_shape, 1, 1, 0, 0.0000001)
    hidden_layer0 = layers.DenseLayer(1 * 6 * 6, 5, relu, relu_d)
    output_layer = layers.DenseLayer(5, 2, sigmoid, sigmoid_d)
    hidden_layers = [hidden_layer0, output_layer]

    nn = network.Network(input_layer, hidden_layers)
    for instance in data:
        print(nn.train(instance[0], instance[1]))
    print(circle)
    print(cross)
    quit()

if __name__ == "__main__":
    main()
