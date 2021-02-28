import layers
import utility
import network
import numpy as np

def main():
    relu = utility.Relu()
    relu_d = utility.MatrixDerivativeRelu()

    sigmoid = utility.Sigmoid()
    sigmoid_d = utility.DerivativeSigmoid()

    data = np.random.rand(8, 8)
    label = np.random.rand(1, 4)

    kernel_n = 2
    kernel_m = 2
    input_channel = 1
    output_channel = 1
    h_stride = 1
    v_stride = 1
    padding = 0
    la = 0.05

    conv_layer0 = layers.ConvolutionalLayer(3, 3, 1, 2, h_stride, v_stride, padding, la)
    conv_layer1 = layers.ConvolutionalLayer(kernel_n, kernel_m, 2, 2, h_stride, v_stride, padding, la)
    conv_layer2 = layers.ConvolutionalLayer(kernel_n, kernel_m, 2, 2, h_stride, v_stride, padding, la)
    dense_layer3 = layers.DenseLayer(8 * 4 * 4, 4, relu, relu_d, la )
    dense_layer4 = layers.DenseLayer(4, 4, sigmoid, sigmoid_d, la )

    input_layer = conv_layer0
    hidden_layers = [conv_layer1, conv_layer2, dense_layer3, dense_layer4]
    nn = network.Network(input_layer, hidden_layers)
    for i in range(10000):
        print(nn.train(label, data).sum())


if __name__ == "__main__":
    main()
