from tkinter import *
import math

class square:
    def __init__(self, radius, pos):
        self.radius = radius
        self.pos = pos

    def belong2circle(self, point):
        return math.pow(point[0] - self.pos[0], 2) + math.pow(point[1] - self.pos[1], 2) <= math.pow(self.radius, 2)


class Selection:
    def __init__(self, color=(255,255,255)):
        self.color = color
        self.image = -1

    def set_startpos(self, startpos):
        self.x = startpos[0]
        self.y = startpos[1]
        self.newx = self.x
        self.newy = self.y

    def set_newpos(self, newpos):
        self.prevx = self.newx
        self.prevy = self.newy
        self.newx = newpos[0]
        self.newy = newpos[1]

    def get_top_left(self):
        return [min(self.x, self.newx), min(self.y, self.newy)]

    def get_botton_right(self):
        return [max(self.x, self.newx), max(self.y, self.newy)]

    def get_prev_top_left(self):
        return [min(self.x, self.prevx), min(self.y, self.prevy)]

    def get_prev_botton_right(self):
        return [max(self.x, self.prevx), max(self.y, self.prevy)]

    def set_image(self, id):
        self.image = id