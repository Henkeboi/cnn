import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def show_figure(figure):
    plt.figure()
    node_counter = 0
    pos = {}
    edges = []
    colors = []
    for x in range(0, figure.shape[1]):
        for y in range(0, figure.shape[0]):
            pos[node_counter] = (x, y)
            if figure[y][x] == 1.0:
                colors.append('blue')
            elif figure[y][x] == 0.0:
                colors.append('red')
            node_counter = node_counter + 1
    G = nx.Graph()
    G.add_nodes_from(pos.keys())
    nx.draw(G, pos, node_color=colors)
    plt.show(block=False)

def show_loss(loss, title):
    plt.figure()
    plt.plot(loss)
    plt.title(title)
    plt.show(block=False)

def show_hinton(matrix, max_weight=None, ax=None):
    plt.figure()
    ax = ax if ax is not None else plt.gca()
    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size, facecolor=color, edgecolor=color)

        ax.add_patch(rect)
    ax.autoscale_view()
    ax.invert_yaxis()
    plt.show(block=False)

