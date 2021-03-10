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

    size = 10
    noise = 0.0
    shapes = ["circle", "cross", "rectangle", "triangle"]
    generator = Generator(size, noise, shapes)
    data = generator.generate(2000)
    
    input_shape = (size, size, 1)
    kernel_n = 4
    kernel_m = 4
    channels_out = 2
    output_shape = (kernel_n, kernel_m, channels_out)
    input_layer = layers.ConvolutionLayer(input_shape, output_shape, 1, 1, 0, 0.01)
    #input_layer = layers.InputLayer(size * size, 25)
    hidden_layer0 = layers.DenseLayer(input_layer.get_output_shape(), 5, relu, relu_d, 0.001)
    output_layer = layers.DenseLayer(hidden_layer0.get_output_shape(), len(shapes), softmax, softmax_d, 0.001)
    hidden_layers = [hidden_layer0, output_layer]

    nn = network.Network(input_layer, hidden_layers)
    for instance in data:
        #print(nn.train(instance[0], instance[1]))
        nn.train(instance[0], instance[1])


    circle = generator.generate_circle()
    prediction = nn.forward_pass(circle)[1][-1]
    #generator.show(circle)
    print("Circle prediction:")
    print(prediction)

    cross = generator.generate_cross()
    prediction = nn.forward_pass(cross)[1][-1]
    #generator.show(cross)
    print("Cross prediction:")
    print(prediction)

    rectangle = generator.generate_rectangle()
    prediction = nn.forward_pass(rectangle)[1][-1]
    #generator.show(rectangle)
    print("Rectangle prediction:")
    print(prediction)

    triangle = generator.generate_triangle()
    prediction = nn.forward_pass(triangle)[1][-1]
    #generator.show(triangle)
    print("Triangle prediction:")
    print(prediction)



if __name__ == "__main__":
    main()
