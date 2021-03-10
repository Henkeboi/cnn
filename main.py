import layers
import utility
import network
import visualization
from generator import Generator
import numpy as np
import matplotlib.pyplot as plt
import config

def main():
    relu = utility.Relu()
    relu_d = utility.DerivativeRelu()

    sigmoid = utility.Sigmoid()
    sigmoid_d = utility.DerivativeSigmoid()

    softmax = utility.Softmax()
    softmax_d = utility.DerivativeSoftmax()

    mse = utility.MSE()

    generator_config = config.GeneratorConfig()
    size, noise, training_set_size, validation_set_size, shapes = generator_config.get_config().values()

    generator = Generator(size, noise, shapes)
    training_set = generator.generate(training_set_size)

    
    network_config = config.NetworkConfig(size, len(shapes))
    nn = network_config.get_network()
    
    training_loss = []
    for i, (label, figure) in enumerate(training_set):
        training_loss.append(nn.train(label, figure).sum())
        if i % 1000 == 0:
            print("Training samples done: " + str(i))

    plot_figures = True
    if plot_figures:
        if "circle" in shapes:
            circle = generator.generate_circle()
            prediction = nn.forward_pass(circle)[1][-1]
            visualization.show_figure(circle)
            print("Circle prediction:")
            print(prediction)

        if "cross" in shapes:
            cross = generator.generate_cross()
            prediction = nn.forward_pass(cross)[1][-1]
            visualization.show_figure(cross)
            print("Cross prediction:")
            print(prediction)

        if "rectangle" in shapes:
            rectangle = generator.generate_rectangle()
            prediction = nn.forward_pass(rectangle)[1][-1]
            visualization.show_figure(rectangle)
            print("Rectangle prediction:")
            print(prediction)

        if "triangle" in shapes:
            triangle = generator.generate_triangle()
            prediction = nn.forward_pass(triangle)[1][-1]
            visualization.show_figure(triangle)
            print("Triangle prediction:")
            print(prediction)


    plot_loss = True
    if plot_loss:
        validation_set = generator.generate(validation_set_size)
        validation_loss = []
        for label, figure in validation_set:
            prediction = nn.forward_pass(figure)[1][-1]
            validation_loss.append(mse(label, prediction).sum())

        visualization.show_loss(training_loss, "Training loss")
        visualization.show_loss(validation_loss, "Validation loss")
    
    plot_hinton = True
    if plot_hinton:
        for layer in nn.get_layers():
            if type(layer) == layers.ConvolutionLayer:
                for kernel in layer.get_kernels():
                    visualization.show_hinton(kernel, 5)

    plt.pause(1000)    
    

if __name__ == "__main__":
    main()
