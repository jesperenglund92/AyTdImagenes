class ATIImage(object):
    def __init__(self, data=[], width=0, height=0, type=0, topleft=None):
        self.data = data
        self.width = width
        self.height = height
        self.type = type
        self.topleft = topleft

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

    def set_at(self, pos, color):
        self.data[pos[1]][pos[0]] = color
