import numpy as np
import math
import random
import networkx as nx
import matplotlib.pyplot as plt

class Generator:
    def __init__(self, size, noise, shapes):
        self._size = size
        self._noise = noise
        self._data = None
        self._on = 1.0
        self._off = 0.0
        self._noise = noise
        self._display_time = 5
        self._shapes = shapes

        self._circle_label = np.full((1, len(shapes)), 0.0)
        if len(shapes) > 0:
            self._circle_label[0][0] = 1.0

        self._cross_label = np.full((1, len(shapes)), 0.0)
        if len(shapes) > 1:
            self._cross_label[0][1] = 1.0

        self._rectangle_label = np.full((1, len(shapes)), 0.0)
        if len(shapes) > 2:
            self._rectangle_label[0][2] = 1.0


    def show(self, data):
        node_counter = 0
        pos = {}
        edges = []
        colors = []
        for x in range(0, self._size):
            for y in range(0, self._size):
                pos[node_counter] = (x, y)
                if data[y][x] == self._on:
                    colors.append('blue')
                elif data[y][x] == self._off:
                    colors.append('red')
                node_counter = node_counter + 1
        G = nx.Graph()
        G.add_nodes_from(pos.keys())
        plt.clf()
        nx.draw(G, pos, node_color=colors)
        plt.show(block=False)
        plt.pause(self._display_time)

    def generate(self, num):
        data = []
        for i in range(num):
            if "circle" in self._shapes:
                circle = self.generate_circle()
                data.append((self._circle_label, circle))
            if "cross" in self._shapes:
                cross = self.generate_cross()
                data.append((self._cross_label, cross))
            if "rectangle" in self._shapes:
                rectangle = self.generate_rectangle()
                data.append((self._rectangle_label, rectangle))
        return data



    def generate_circle(self):
        circle = np.ones((self._size, self._size), dtype=float) * self._off
        radius = self._size / 3

        for x in range(0, self._size):
            for y in range(0, self._size):
                x_center = x - self._size / 2
                y_center = y - self._size / 2
                if math.sqrt(x_center ** 2 + y_center ** 2) >= radius - 1.5 and math.sqrt(x_center ** 2 + y_center ** 2) <= radius + 1.5:
                    circle[x][y] = self._on
        circle = self._add_noise(circle)
        return circle 

    def generate_cross(self):
        cross = np.ones((self._size, self._size), dtype=float) * self._off
        for x in range(0, self._size):
            for y in range(0, self._size):
                if x == y or x == self._size - y - 1:
                    cross[x][y] = self._on
        cross = self._add_noise(cross)
        return cross

    def generate_rectangle(self):
        rectangle = np.ones((self._size, self._size), dtype=float) * self._off 
        for x in range(0, self._size):
            for y in range(0, self._size):
                if x > int(self._size / 6) and x < int(5 * self._size / 6):
                    if y == int(self._size / 6):
                        rectangle[x][y] = self._on
                    if y == int(5 * self._size / 6):
                        rectangle[x][y] = self._on
                if x == int(self._size / 6) or x == int(5 * self._size / 6):
                    if y >= int(self._size / 6) and y <= int(5 * self._size / 6):
                        rectangle[x][y] = self._on
        rectangle = self._add_noise(rectangle)
        return rectangle

 
    def _add_noise(self, frame):
        is_noisy_pixel = lambda : random.random() < self._noise
        for x in range(0, np.shape(frame)[1]):
            for y in range(0, np.shape(frame)[0]):
                if is_noisy_pixel():
                    if frame[x][y] == self._off:
                        frame[x][y] = self._on
                    else:
                        frame[x][y] = self._off
        return frame 
