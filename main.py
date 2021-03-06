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

    size = 20
    noise = 0.1
    shapes = ["circle", "cross"]
    generator = Generator(size, noise, shapes)
    data = generator.generate(50)
    
    input_shape = (size, size, 1)
    kernel_n = 4
    kernel_m = 4
    channels_out = 1
    output_shape = (kernel_n, kernel_m, channels_out)
    input_layer = layers.ConvolutionalLayer(input_shape, output_shape, 1, 1, 0, 0.01)
    hidden_layer0 = layers.DenseLayer(input_layer.get_output_shape(), 5, relu, relu_d, 0.01)
    output_layer = layers.DenseLayer(hidden_layer0.get_output_shape(), len(shapes), sigmoid, sigmoid_d, 0.01)
    hidden_layers = [hidden_layer0, output_layer]

    nn = network.Network(input_layer, hidden_layers)
    for instance in data:
        #print(nn.train(instance[0], instance[1]))
        nn.train(instance[0], instance[1])


    circle = generator.generate_circle()
    prediction = nn.forward_pass(circle)[1][-1]
    generator.show(circle)
    print("Circle prediction:")
    print(prediction)

    cross = generator.generate_cross()
    prediction = nn.forward_pass(cross)[1][-1]
    generator.show(cross)
    print("Cross prediction:")
    print(prediction)





if __name__ == "__main__":
    main()
