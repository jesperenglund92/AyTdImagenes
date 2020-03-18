from tkinter import *
import math

class square:
    def __init__(self, radius, pos):
        self.radius = radius
        self.pos = pos

    def belong2circle(self, point):
        return math.pow(point[0] - self.pos[0], 2) + math.pow(point[1] - self.pos[1], 2) <= math.pow(self.radius, 2)


class Selection:
    def __init__(self, data=None):
        self.data = data
        self.color = (0, 0, 255)

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
