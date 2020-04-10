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
        self.tlx_i = 0
        self.tly_i = 0
        self.brx_i = 0
        self.bry_i = 0
        self.points_outside = []

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

    def get_top_right(self):
        return [max(self.x, self.new_x), min(self.y, self.new_y)]

    def get_bottom_right(self):
        return [max(self.x, self.new_x), max(self.y, self.new_y)]

    def get_bottom_left(self):
        return [min(self.x, self.new_x), max(self.y, self.new_y)]

    def get_prev_top_left(self):
        return [min(self.x, self.prev_x), min(self.y, self.prev_y)]

    def get_prev_bottom_right(self):
        return [max(self.x, self.prev_x), max(self.y, self.prev_y)]

    def set_image(self, image_id):
        self.image = image_id

    def get_width(self):
        tl = self.get_top_left()
        br = self.get_bottom_right()
        return br[0] - tl[0] + 1

    def get_height(self):
        tl = self.get_top_left()
        br = self.get_bottom_right()
        return br[1] - tl[1] + 1

    @staticmethod
    def get_pixel_count(data):
        if len(data) > 0:
            return str(len(data)) + " x " + str(len(data[0]))
        else:
            return "0"

    def get_image_within_selection(self):
        return (self.tlx_i, self.tly_i), (self.brx_i, self.bry_i)

    def set_image_within_selection(self, i_tl, i_br, i_width, i_height):
        tlx, tly = self.get_prev_top_left()
        brx, bry = self.get_prev_bottom_right()
        if tlx >= i_tl[0] + i_width:
            tlx = i_tl[0] + i_width - 1
        if tlx < i_tl[0]:
            tlx = i_tl[0]
        if tly >= i_tl[1] + i_height:
            tly = i_tl[1] + i_height - 1
        if tly < i_tl[1]:
            tly = i_tl[1]
        if brx <= i_br[0] - i_width:
            brx = i_br[0] - i_width + 1
        if brx > i_br[0]:
            brx = i_br[0]
        if bry <= i_br[1] - i_height:
            bry = i_br[1] - i_height + 1
        if bry > i_br[1]:
            bry = i_br[1]
        self.tlx_i = tlx
        self.tly_i = tly
        self.brx_i = brx
        self.bry_i = bry



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
