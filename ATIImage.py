import math

class ATIImage(object):
    def __init__(self, data=None, width=0, height=0, type=0, topleft=None, editable=False, values_set=False):
        self.data = data
        self.width = width
        self.height = height
        self.type = type
        self.topleft = topleft
        self.editable = editable #use these sort of attributes to separate different images from eachother when iterating through "images" list
        self.values_set = values_set

    """def draw(self):
        for x in range(self.height):
            for y in range(self.width):
                self.surface.set_at((x + self.topleft[0], y + self.topleft[0]), self.data[x][y])"""

    def get_red_band(self):
        data = []
        for y in range(self.height):
            tmpList = []
            for x in range(self.width):
                tmpList.append(self.data[x][y][0])
            data.append(tmpList)
        return data

    def get_blue_band(self):
        data = []
        for y in range(self.height):
            tmpList = []
            for x in range(self.width):
                tmpList.append(self.data[x][y][2])
            data.append(tmpList)
        return data

    def get_green_band(self):
        data = []
        for y in range(self.height):
            tmpList = []
            for x in range(self.width):
                tmpList.append(self.data[x][y][1])
            data.append(tmpList)
        return data

    def get_at(self, pos):
        return self.data[pos[1]][pos[0]]

    def get_at_screenpos(self, x, y):
        # get colorvalue based on a screen position
        return self.data[y - self.topleft[1]][x - self.topleft[0]]

    def set_at(self, pos, color):
        self.data[pos[1]][pos[0]] = color

    def collidepoint(self, x, y):
        # check if arguments x and why "collides" on image
        if self.values_set:
            if self.topleft[1] < x < self.topleft[1] + self.width and self.topleft[0] < y < self.topleft[0] + self.height:
                return True

def rgbcolor2hsvcolor(rgbdata):
    r = rgbdata[0]
    g = rgbdata[1]
    b = rgbdata[2]

    maxcolor = max(r, g, b)
    mincolor = min(r, g, b)

    s, v = 0, 0

    if maxcolor == mincolor:
        h = "n/a"

    if maxcolor == r:
        if g >= b:
            h = round(60 * (g - b) / (maxcolor - mincolor)) % 360
        else:
            h = (round(60 * (g - b) / (maxcolor - mincolor)) + 360) % 360

    if maxcolor == g:
        h = (round(60 * (b - r) / (maxcolor - mincolor)) + 120) % 360
    if maxcolor == b:
        h = (round(60 * (r - g) / (maxcolor - mincolor)) + 240) % 360

    if maxcolor == 0:
        s = 0
    else:
        s = 1 - mincolor / maxcolor

    v = maxcolor
    return [h, s, v]

def rgb2hsv(imageData, width, height):
    hsvData = []
    for y in range(height):
        tmpList = []
        for x in range(width):
            tmpList.append(rgbcolor2hsvcolor(imageData[x][y]))
        hsvData.append(tmpList)
    return hsvData

def hsvcolor2rgbcolor(hsvdata):
    h = hsvdata[0]
    s = hsvdata[1]
    v = hsvdata[2]

    if h > 360:
        h = math.fmod(h, 360)

    auxH = math.ceil(h / 60) % 6
    f = ((math.ceil(h) / 60) % 6) - auxH
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if auxH == 0:
        return [v, t, p]
    if auxH == 1:
        return [q, v, p]
    if auxH == 2:
        return [p, v, t]
    if auxH == 3:
        return [p, q, v]
    if auxH == 4:
        return [t, p, v]
    if auxH == 5:
        return [v, p, q]
    return "n/a"

def hsv2rgb(imageData, width, height):
    rgbData = []
    for y in range(height):
        tmpList = []
        for x in range(width):
            tmpList.append(hsvcolor2rgbcolor(imageData[x][y]))
        rgbData.append(tmpList)
    return rgbData
