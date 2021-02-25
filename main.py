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
    #label = np.full((1, 3), 1.0)


    kernel_n = 1
    kernel_m = 2
    input_channel = 1
    output_channel = 1
    h_stride = 1
    v_stride = 1
    padding = 0

    input_layer = layers.InputLayer(25, 10)
    conv_layer0 = layers.ConvolutionalLayer(kernel_n, kernel_m, 1, 2, h_stride, v_stride, padding)
    conv_layer1 = layers.ConvolutionalLayer(kernel_n, kernel_m, 2, 6, h_stride, v_stride, padding)


    data0 = input_layer.forward(data)
    data1 = conv_layer0.forward(data0)
    data2 = conv_layer1.forward(data1)

    quit()











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
