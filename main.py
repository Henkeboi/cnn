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

    size = 16
    noise = 0.1
    generator = Generator(size, noise)
    data = generator.generate(50)
    generator.show(data)
    quit()
    
    input_shape = (size, size, 1)
    kernel_n = 4
    kernel_m = 4
    channels_out = 2
    output_shape = (kernel_n, kernel_m, channels_out)
    input_layer = layers.ConvolutionalLayer(input_shape, output_shape, 1, 1, 0, 0.01)
    hidden_layer0 = layers.DenseLayer(input_layer.get_output_shape(), 5, relu, relu_d, 0.01)
    output_layer = layers.DenseLayer(hidden_layer0.get_output_shape(), 2, sigmoid, sigmoid_d, 0.01)
    hidden_layers = [hidden_layer0, output_layer]

    nn = network.Network(input_layer, hidden_layers)
    for instance in data:
        print(nn.train(instance[0], instance[1]))

    print(circle)
    print(cross)
    quit()

if __name__ == "__main__":
    main()
