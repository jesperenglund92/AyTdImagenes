from tkinter import *
import math
import random
import numpy


class Square:
    def __init__(self, radius, pos):
        self.radius = radius
        self.pos = pos

    def belong2circle(self, point):
        return math.pow(point[0] - self.pos[0], 2) + math.pow(point[1] - self.pos[1], 2) <= math.pow(self.radius, 2)


class Selection:
    def __init__(self, color=(255, 255, 255)):
        self.color = color
        self.image = -1
        self.x = 0
        self.y = 0
        self.new_x = self.x
        self.new_y = self.y
        self.prev_x = 0
        self.prev_y = 0

    def set_start_pos(self, start_pos):
        self.x = start_pos[0]
        self.y = start_pos[1]
        self.new_x = self.x
        self.new_y = self.y

    def set_new_pos(self, new_pos):
        self.prev_x = self.new_x
        self.prev_y = self.new_y
        self.new_x = new_pos[0]
        self.new_y = new_pos[1]

    def get_top_left(self):
        return [min(self.x, self.new_x), min(self.y, self.new_y)]

    def get_botton_right(self):
        return [max(self.x, self.new_x), max(self.y, self.new_y)]

    def get_prev_top_left(self):
        return [min(self.x, self.prev_x), min(self.y, self.prev_y)]

    def get_prev_botton_right(self):
        return [max(self.x, self.prev_x), max(self.y, self.prev_y)]

    def set_image(self, image_id):
        self.image = image_id

    def get_width(self):
        tl = self.get_top_left()
        br = self.get_botton_right()
        return br[0] - tl[0] + 1

    def get_height(self):
        tl = self.get_top_left()
        br = self.get_botton_right()
        return br[1] - tl[1] + 1

    def get_pixel_count(self):
        return self.get_width().__str__() + " x " + self.get_height().__str__()


class ATIRandom:
    def __init__(self):
        pass

    @classmethod
    def random(cls):
        return random.random()

    @classmethod
    def gaussian(cls, sigma, mu):
        return numpy.random.normal(float(mu), float(sigma))

    @classmethod
    def exponential(cls, gamma):
        return numpy.random.exponential(float(gamma))

    @classmethod
    def rayleigh(cls, epsilon):
        return numpy.random.rayleigh(float(epsilon))

    @classmethod
    def has_to_apply(cls, value):
        return cls.random() <= value
