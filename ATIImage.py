import math


class ATIImage(object):
    def __init__(self, data=None, width=0, height=0, type=0, topleft=None, active=False):
        if data is None:
            data = []
        self.data = data
        self.width = width
        self.height = height
        self.type = type
        self.topleft = topleft
        self.active = active

    def image_color_type(self):
        if self.type == '.raw' or self.type == '.pgm':
            return 'g'
        return 'rgb'

    def get_top_left(self):
        return self.topleft

    def get_botton_right(self):
        return [self.topleft[0] + self.width, self.topleft[1] + self.height]

    def in_display_image(self, pos):
        botton_right = self.get_botton_right()
        return self.topleft[0] <= pos[0] <= botton_right[0] and self.topleft[1] <= pos[1] <= botton_right[1]

    def set_top_left(self, pos):
        self.topleft = pos

    def get_red_band(self):
        data = []
        for y in range(self.height):
            tmp_list = []
            for x in range(self.width):
                tmp_list.append(self.data[x][y][0])
            data.append(tmp_list)
        return data

    def get_blue_band(self):
        data = []
        for y in range(self.height):
            tmp_list = []
            for x in range(self.width):
                tmp_list.append(self.data[x][y][2])
            data.append(tmp_list)
        return data

    def get_green_band(self):
        data = []
        for y in range(self.height):
            tmp_list = []
            for x in range(self.width):
                tmp_list.append(self.data[x][y][1])
            data.append(tmp_list)
        return data

    def get_at(self, pos):
        return self.data[pos[1]][pos[0]]

    def get_at_display(self, pos):
        x = pos[0] - self.topleft[0]
        y = pos[1] - self.topleft[1]
        if not (0 <= x <= self.width - 1):
            raise Exception("Invalid position")
        if not (0 <= y <= self.height - 1):
            raise Exception("Invalid position")
        return self.get_at((x, y))

    def set_at_display(self, pos, color):
        x = pos[0] - self.topleft[0]
        y = pos[1] - self.topleft[1]
        if not (0 <= x <= self.width - 1):
            raise Exception("Invalid position")
        if not (0 <= y <= self.height - 1):
            raise Exception("Invalid position")
        self.set_at((x, y), color)

    def __get_band_average_display(self, tl, br, band):
        left = tl[0]
        top = tl[1]
        right = br[0]
        botton = br[1]
        count = 0
        total = 0
        for x in range(right - left + 1):
            for y in range(botton - top + 1):
                count = count + 1
                total = total + self.get_at_display((x + left, y + top))[band]
        return round(total / count, 2)

    def get_grey_average_display(self, tl, br):
        return self.get_red_average_display(tl, br)

    def get_red_average_display(self, tl, br):
        return self.__get_band_average_display(tl, br, 0)

    def get_green_average_display(self, tl, br):
        return self.__get_band_average_display(tl, br, 1)

    def get_blue_average_display(self, tl, br):
        return self.__get_band_average_display(tl, br, 2)

    def set_at(self, pos, color):
        self.data[pos[1]][pos[0]] = color

    # Add Images
    def add_image(self, image):
        if self.width != image.width or self.height != image.height:
            raise Exception('Image should be same width and height')
        for x in range(self.width):
            for y in range(self.height):
                self.set_at((x, y), [
                    self.get_at((x, y))[0] + image.get_at((x, y))[0],
                    self.get_at((x, y))[1] + image.get_at((x, y))[1],
                    self.get_at((x, y))[2] + image.get_at((x, y))[2]
                ])

        # Todo: Normalizar
        return

    @classmethod
    def add_image(cls, img1, img2):
        if img1.width != img2.width or img1.height != img2 != img2.height:
            raise Exception('Image should be same width and height')
        image = []
        for x in range(img1.width):
            tmp_list = []
            for y in range(img1.height):
                tmp_list.append([
                    img1.get_at((x, y))[0] + img2.get_at((x, y))[0],
                    img1.get_at((x, y))[1] + img2.get_at((x, y))[1],
                    img1.get_at((x, y))[2] + img2.get_at((x, y))[2]
                ])
            image.append(tmp_list)
        return cls(image, img1.width, img2.height, img1.type, img1.topLeft)

    # Subtract images
    def subtract_image(self, image):
        if self.width != image.width or self.height != image.height:
            raise Exception('Image should be same width and height')
        for x in range(self.width):
            for y in range(self.height):
                self.set_at((x, y), [
                    self.get_at((x, y))[0] - image.get_at((x, y))[0],
                    self.get_at((x, y))[1] - image.get_at((x, y))[1],
                    self.get_at((x, y))[2] - image.get_at((x, y))[2]
                ])
        # Todo: Normalizar
        return

    @classmethod
    def subtract_image(cls, img1, img2):
        if img1.width != img2.width or img1.height != img2 != img2.height:
            raise Exception('Image should be same width and height')
        image = []
        for x in range(img1.width):
            tmpList = []
            for y in range(img1.height):
                tmpList.append([
                    img1.get_at((x, y))[0] - img2.get_at((x, y))[0],
                    img1.get_at((x, y))[1] - img2.get_at((x, y))[1],
                    img1.get_at((x, y))[2] - img2.get_at((x, y))[2]
                ])
            image.append(tmpList)
        return cls(image, img1.width, img2.height, img1.type, img1.topLeft)

    # Subtract images
    def multiply_image(self, image):
        # Todo: Implement
        raise Exception('Not implemented method')
        # Verify it is matrix multiplication

        """if self.width != image.width or self.height != image.height:
            raise Exception('Image should be same width and height')
        for x in range(self.width):
            for y in range(self.height):
                self.set_at((x, y), self.get_at((x, y)) - image.get_at((x, y)))

        # Todo: Normalizar
        return"""

    @classmethod
    def multiply_image(cls, img1, img2):
        # Todo: Implement
        raise Exception('Not implemented method')

        """if img1.width != img2.width or img1.height != img2 != img2.height:
            raise Exception('Image should be same width and height')
        image = []
        for x in range(img1.width):
            tmpList = []
            for y in range(img1.height):
                tmpList.append([
                    img1.get_at((x, y))[0] - img2.get_at((x, y))[0],
                    img1.get_at((x, y))[1] - img2.get_at((x, y))[1],
                    img1.get_at((x, y))[2] - img2.get_at((x, y))[2]
                ])
            image.append(tmpList)
        return cls(image, img1.width, img2.height, img1.type, img1.topLeft)
        """

    # Scalar product
    def scalar_product(self, scalar):
        if not isinstance(scalar, int):
            raise Exception('scalar is not an integer value')
        for x in range(self.width):
            for y in range(self.height):
                self.set_at((x, y), self.__scalar_product(self.get_at((x, y)), scalar))

    def __scalar_product(self, array, scalar):
        for x in range(3):
            array[x] = array[x] * scalar
        return array

    # Threshold Function
    def __threshold_assign(self, array, threshold):
        for x in range(3):
            if array[x] <= threshold:
                array[x] = 0
            else:
                array[x] = 255
        return array

    def threshold_function(self, threshold):
        if not isinstance(threshold, int):
            raise Exception('threshold is not an integer value')
        for x in range(self.width):
            for y in range(self.height):
                self.set_at((x, y), self.__threshold_assign(self.get_at((x, y)), threshold))

    # Dynamic compretion
    def dynamic_compression(self):
        # T(r) = c * log( 1 + r )
        # c = L - 1 / log(1 + R)
        # R es el max(r)
        # L = 256

        raise Exception("Not implemented method")
        pass

    def gamma_function(self, gamma):
        raise Exception("Not implemented method")
        # T = c * r ^ gamma
        # L = 256
        # Gamma cant be 0, 1 or 2
        # c = (L - 1)^(1 - gamma)

    # Scalling function
    def normalize_image(self):
        raise Exception("Not implemented method")
        # X' = a + (X - Xmin) * ( b - a) / (Xmax - Xmin)

    def equalize_image(self):
        raise Exception("Not Implementd method")

    def color_array(self):
        array = [None] * 256
        for x in range(self.width):
            for y in range(self.height):
                array[self.get_at((x, y))] = array[self.get_at((x, y))] + 1
        return array

    def negative(self):
        for x in range(self.width):
            for y in range(self.height):
                for z in range(3):
                    self.set_at((x, y), 255 - self.get_at((x, y))[z])


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
