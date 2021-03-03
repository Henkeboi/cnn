import numpy as np
import math
import random

class Generator:
    def __init__(self, size, noise):
        self._size = size
        self._noise = noise
        self._data = None
        self._on = 1.0
        self._off = 0.0
        self._noise = noise

    def generate(self, num):
        circle = self._generate_circle(num).flatten()
        cross = self._generate_cross(num).flatten()
        rectangle = self._generate_rectangle(num).flatten()
        return circle, cross, rectangle

    def _generate_circle(self, num):
        circle = np.ones((self._size, self._size), dtype=float) * self._off
        radius = self._size / 3

        for x in range(0, self._size):
            for y in range(0, self._size):
                x_center = x - self._size / 2
                y_center = y - self._size / 2
                if math.sqrt(x_center ** 2 + y_center ** 2) >= radius - 2.5 and math.sqrt(x_center ** 2 + y_center ** 2) <= radius + 2.5:
                    circle[x][y] = self._on
        circle = self._add_noise(circle)
        return circle 

    def _generate_cross(self, num):
        cross = np.ones((self._size, self._size), dtype=float) * self._off
        for x in range(0, self._size):
            for y in range(0, self._size):
                if x == y or x == self._size - y - 1:
                    cross[x][y] = self._on
        cross = self._add_noise(cross)
        return cross

    def _generate_rectangle(self, num):
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
