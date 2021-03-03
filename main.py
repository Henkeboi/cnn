import layers
import utility
import network
import numpy as np
import math
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

    circle = generator._generate_circle(1)
    cross = generator._generate_cross(1)

    label_circle = np.full((1, 2), 0.0)
    label_circle[0][0] = 1.0
    label_circle[0][1] = 0.0

    label_cross = np.full((1, 2), 0.0)
    label_cross[0][0] = 0.0
    label_cross[0][1] = 1.0



    # input_layer = layers.InputLayer(size ** 2, 30)

    # input_shape0 = (1, 30, 1)
    # output0 = (1, 10, 1)
    # c_layer0 = layers.ConvolutionLayer(input_shape0, output0, 1, 1, 0, 0.000000001)

    # output1 = 16
    # dense_layer1 = layers.DenseLayer(c_layer0.get_output_shape(), output1, relu, relu_d, 0.000001)

    # output2 = 2
    # dense_layer2 = layers.DenseLayer(dense_layer1.get_output_shape(), output2, softmax, sigmoid_d, 0.000001)

    # hidden_layers = [c_layer0, dense_layer1, dense_layer2]
    # nn = network.Network(input_layer, hidden_layers)
    # for i in range(10000):
    #     for _ in range(1):
    #         nn.train(label_circle, circle).sum()

    #     for _ in range(1):
    #         nn.train(label_cross, cross).sum()

    # quit()

    data = np.random.rand(8, 8)
    label = np.random.rand(1, 4)

    kernel_n = 2
    kernel_m = 2
    h_stride = 1
    v_stride = 1
    padding = 0
    la = 0.1

    input_shape0 = (data.shape[0], data.shape[1], 1)
    output_shape0 = (3, 3, 2)
    conv_layer0 = layers.ConvolutionLayer(input_shape0, output_shape0, h_stride, v_stride, padding, la)

    input_shape1 = conv_layer0.get_output_shape() 
    output_shape1 = (kernel_n, kernel_m, 2)
    conv_layer1 = layers.ConvolutionLayer(input_shape1, output_shape1, h_stride, v_stride, padding, la)

    input_shape2 = conv_layer1.get_output_shape()
    output_shape2 = (2, 2, 2)
    conv_layer2 = layers.ConvolutionLayer(input_shape2, output_shape2, h_stride, v_stride, padding, la)

    dense_layer3 = layers.DenseLayer(conv_layer2.get_output_shape(), 4, relu, relu_d, la)
    dense_layer4 = layers.DenseLayer(dense_layer3.get_output_shape(), 4, sigmoid, sigmoid_d, la)

    input_layer = conv_layer0
    hidden_layers = [conv_layer1, conv_layer2, dense_layer3, dense_layer4]
    nn = network.Network(input_layer, hidden_layers)
    for i in range(1000):
        nn.train(label, data).sum()


if __name__ == "__main__":
    main()
