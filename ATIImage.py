import math
import copy
from classes import *

class ATIImage(object):
    def __init__(self, data=None, width=0, height=0, type=0, topleft=None, active=False, editable=False,
                 values_set=False):
        if data is None:
            data = []
        self.data = data
        self.width = width
        self.height = height
        self.type = type
        self.topleft = topleft
        self.editable = editable  # use these sort of attributes to separate different images from eachother when iterating through "images" list
        self.values_set = values_set
        self.active = active

    def get_copy(self):
        new_data = self.__copy_data()
        new_width = copy.copy(self.width)
        new_height = copy.copy(self.height)
        new_type = copy.copy(self.type)
        new_tl = copy.copy(self.topleft)
        new_activate = copy.copy(self.active)
        new_editable = copy.copy(self.editable)
        new_value_set = copy.copy(self.values_set)

        copy_image = ATIImage(new_data, new_width, new_height,
                              new_type, new_tl, new_activate, new_editable,
                              new_value_set)
        # copy_image = copy.deepcopy(self)
        return copy_image

    def __copy_data(self):
        new_data = []
        for y in range(self.height):
            tmpRow = []
            for x in range(self.width):
                tmpRow.append(copy.copy(self.get_at((x, y))))
            new_data.append(tmpRow)
        return new_data

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

    def get_pos_display(self, pos):
        return pos[0] - self.topleft[0], pos[1] - self.topleft[1]

    def get_at_display(self, pos):
        x = pos[0] - self.topleft[0]
        y = pos[1] - self.topleft[1]
        if not (0 <= x <= self.width):
            raise Exception("Invalid position")
        if not (0 <= y <= self.height):
            raise Exception("Invalid position")
        return self.get_at((x - 1, y - 1))

    def set_at_display(self, pos, color):
        x = pos[0] - self.topleft[0] - 1
        y = pos[1] - self.topleft[1] - 1
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

    def get_at_screenpos(self, x, y):
        # get colorvalue based on a screen position
        return self.data[y - self.topleft[1]][x - self.topleft[0]]

    def set_at(self, pos, color):
        self.data[pos[1]][pos[0]] = color

    def set_at_band(self, pos, color, band):
        self.data[pos[1]][pos[0]][band] = color

    """
    
    Noises
    
    """
    def noise_gaussian(self, percent, mu=0, sigma=1):
        # Gaussian is Aditive
        for x in range(self.width):
            for y in range(self.height):
                if ATIRandom.has_to_apply(percent):
                    variation = ATIRandom.gaussian(sigma=sigma, mu= mu)
                    self.set_at_band((x, y), self.get_at(x, y)[0] + variation, 0)
                    self.set_at_band((x, y), self.get_at(x, y)[1] + variation, 1)
                    self.set_at_band((x, y), self.get_at(x, y)[2] + variation, 2)
        return

    def noise_rayleigh(self, percent, epsilon):
        # Rayleigh is Multiplicative
        for x in range(self.width):
            for y in range(self.height):
                if ATIRandom.has_to_apply(percent):
                    variation = ATIRandom.rayleigh(epsilon)
                    self.set_at_band((x, y), self.get_at(x, y)[0] * variation, 0)
                    self.set_at_band((x, y), self.get_at(x, y)[1] * variation, 1)
                    self.set_at_band((x, y), self.get_at(x, y)[2] * variation, 2)
        return

    def noise_exponential(self, percent, gamma):
        # Rayleigh is Multiplicative
        for x in range(self.width):
            for y in range(self.height):
                if ATIRandom.has_to_apply(percent):
                    variation = ATIRandom.exponential(gamma)
                    self.set_at_band((x, y), self.get_at(x, y)[0] * variation, 0)
                    self.set_at_band((x, y), self.get_at(x, y)[1] * variation, 1)
                    self.set_at_band((x, y), self.get_at(x, y)[2] * variation, 2)
        return

    def noise_salt_and_pepper(self, density):
        # Rayleigh is Multiplicative
        for x in range(self.width):
            for y in range(self.height):
                random = ATIRandom.random()
                if random <= density:
                    self.set_at((x, y), [0, 0, 0])
                if random >= (1 - p):
                    self.set_at((x, y), [255, 255, 255])
        return

    """@classmethod
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
        return cls(image, img1.width, img2.height, img1.type, img1.topLeft)"""


    """
    
    Image Operations
    
    """

    # Add Images
    def add_image(self, image):
        if self.width != image.width or self.height != image.height:
            raise Exception('Image should be same width and height')
        for x in range(self.width):
            for y in range(self.height):
                color1 = self.get_at((x, y))
                color2 = image.get_at((x, y))
                new = [color1[0] + color2[0],
                       color1[1] + color2[1],
                       color1[2] + color2[2]]

                self.set_at((x, y), [
                    self.get_at((x, y))[0] + image.get_at((x, y))[0],
                    self.get_at((x, y))[1] + image.get_at((x, y))[1],
                    self.get_at((x, y))[2] + image.get_at((x, y))[2]
                ])

        if self.__needs_compression():
            self.__scalling_compression()
        return

    def __needs_compression(self):
        return self.__need_compression_band(0) or self.__need_compression_band(1) or self.__need_compression_band(2)

    def __need_compression_band(self, band):
        val_min_band, val_max_band = self.__get_min_max_by_band(band)
        if not (0 <= val_min_band <= val_max_band <= 255):
            return True
        return False

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
        self.__scalling_compression()
        return

    # @classmethod
    # def subtract_image(cls, img1, img2):
    #    if img1.width != img2.width or img1.height != img2 != img2.height:
    #        raise Exception('Image should be same width and height')
    #    image = []
    #    for x in range(img1.width):
    #        tmpList = []
    #        for y in range(img1.height):
    #            tmpList.append([
    #                img1.get_at((x, y))[0] - img2.get_at((x, y))[0],
    #                img1.get_at((x, y))[1] - img2.get_at((x, y))[1],
    #                img1.get_at((x, y))[2] - img2.get_at((x, y))[2]
    #            ])
    #        image.append(tmpList)
    #    return cls(image, img1.width, img2.height, img1.type, img1.topLeft)

    # Subtract images
    def multiply_image(self, image):
        if image.width != self.width or image.height != self.height:
            raise Exception('Image should be same width and height')

        for x in range(self.width):
            for y in range(self.height):
                color1 = self.get_at((x, y))
                color2 = image.get_at((x, y))
                ans = [color1[0] * color2[0],
                       color1[1] * color2[1],
                       color1[2] * color2[2]]
                self.set_at((x, y), ans)

        self.__scalling_compression()

    """@classmethod
    def multiply_image(cls, img1, img2):
        # Todo: Implement
        raise Exception('Not implemented method')
    """
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
        self.dynamic_compression()

    def __scalar_product(self, array, scalar):
        for x in range(3):
            array[x] = array[x] * scalar
        return array

    # Threshold Function
    def __threshold_assign(self, array, threshold):
        # for x in range(3):
        if array[0] <= threshold:
            array[0] = 0
            array[1] = 0
            array[2] = 0
        else:
            array[0] = 255
            array[1] = 255
            array[2] = 255
        return array

    def threshold_function(self, threshold):
        if not isinstance(threshold, int):
            raise Exception('threshold is not an integer value')
        for x in range(self.width):
            for y in range(self.height):
                self.set_at((x, y), self.__threshold_assign(self.get_at((x, y)), threshold))

    def __scalling_compression(self):
        self.__scalling_compression_by_band(0)
        self.__scalling_compression_by_band(1)
        self.__scalling_compression_by_band(2)

    def fill_band_black(self, band):
        for x in range(self.width):
            for y in range(self.height):
                self.set_at_band((x, y), 0, band)

    def __scalling_compression_by_band(self, band):
        min_val = self.__get_min_by_band(band)
        max_val = self.__get_max_by_band(band)
        if min_val == max_val:
            if min_val < 0 or min_val > 255:
                self.fill_band_black(band)
            return
        b = 255
        for x in range(self.width):
            for y in range(self.height):
                prev = self.get_at((x, y))[band]
                new = (prev - min_val) * b / (max_val - min_val)
                new = int(math.floor(new))
                self.set_at_band((x, y), new, band)

    def __get_min_max_by_band(self, band):
        min_val = self.get_at((0, 0))[band]
        max_val = self.get_at((0, 0))[band]
        for x in range(self.width):
            for y in range(self.height):
                num = self.get_at((x, y))[band]
                if num < min_val:
                    min_val = num
                if num > max_val:
                    max_val = num
        return min_val, max_val

    def __get_max_by_band(self, band):
        max_val = self.get_at((0, 0))[band]
        for x in range(self.width):
            for y in range(self.height):
                num = self.get_at((x, y))[band]
                if num > max_val:
                    max_val = num
        return max_val

    def __get_min_by_band(self, band):
        min_val = self.get_at((0, 0))[band]
        for x in range(self.width):
            for y in range(self.height):
                num = self.get_at((x, y))[band]
                if num < min_val:
                    min_val = num
        return min_val

    # Dynamic compretion
    def dynamic_compression(self):
        self.__dynamic_compression_band(0)
        self.__dynamic_compression_band(1)
        self.__dynamic_compression_band(2)
        pass

    def __dynamic_compression_band(self, band):
        # T(r) = c * log( 1 + r )
        # c = L - 1 / log(1 + R)
        # R es el max(r)
        # L = 256
        max_val = self.__get_max_by_band(band)
        # min_val, max_val = self.__get_min_max_by_range(band)
        l = 256

        for x in range(self.width):
            for y in range(self.height):
                prev = self.get_at((x, y))[band]
                new = ((l - 1) * math.log10(1 + prev)) / math.log10(1 + max_val)
                new = int(round(new))
                self.set_at_band((x, y), new, band)

    def gamma_function(self, gamma):
        self.__gamma_function_by_band(gamma, 0)
        self.__gamma_function_by_band(gamma, 1)
        self.__gamma_function_by_band(gamma, 2)

    def __gamma_function_by_band(self, gamma, band):
        # T = c * r ^ gamma
        # L = 256
        # Gamma cant be 0, 1 or 2
        # c = (L - 1)^(1 - gamma)
        l = 256
        for x in range(self.width):
            for y in range(self.height):
                prev = self.get_at((x, y))[band]
                new = math.pow((l - 1), (1 - gamma)) * math.pow(prev, gamma)
                new = int(math.floor(new))
                self.set_at_band((x, y), new, band)

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
                color = self.get_at((x, y))
                self.set_at((x, y), (255 - color[0], 255 - color[1], 255 - color[2]))


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


def rgb2hsv(image_data, width, height):
    hsvData = []
    for y in range(height):
        tmpList = []
        for x in range(width):
            tmpList.append(rgbcolor2hsvcolor(image_data[x][y]))
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
