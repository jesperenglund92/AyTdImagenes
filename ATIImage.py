import copy
from classes import *


class ATIImage(object):
    def __init__(self, data=None, width=0, height=0, image_type=0, top_left=None, active=False, editable=False,
                 values_set=False):
        if data is None:
            data = []
        self.data = data
        self.width = width
        self.height = height
        self.image_type = image_type
        self.top_left = top_left
        # use these sort of attributes to separate different images from each other when iterating through "images" list
        self.editable = editable
        self.values_set = values_set
        self.active = active
        self.magic_num = None
        self.max_gray_level = None

    #
    #   Getters
    #

    def get_copy(self):
        new_data = self.__copy_data()
        new_width = copy.copy(self.width)
        new_height = copy.copy(self.height)
        new_type = copy.copy(self.image_type)
        new_tl = copy.copy(self.top_left)
        new_activate = copy.copy(self.active)
        new_editable = copy.copy(self.editable)
        new_value_set = copy.copy(self.values_set)

        copy_image = ATIImage(new_data, new_width, new_height,
                              new_type, new_tl, new_activate, new_editable,
                              new_value_set)
        copy_image.magic_num = self.magic_num
        copy_image.max_gray_level = self.max_gray_level
        # copy_image = copy.deepcopy(self)
        return copy_image

    def __copy_data(self):
        new_data = []
        for y in range(self.height):
            tmp_row = []
            for x in range(self.width):
                tmp_row.append(copy.copy(self.get_at((x, y))))
            new_data.append(tmp_row)
        return new_data

    def image_color_type(self):
        if self.image_type == '.raw' or self.image_type == '.pgm':
            return 'g'
        return 'rgb'

    def get_top_left(self):
        return self.top_left

    def get_botton_right(self):
        return [self.top_left[0] + self.width, self.top_left[1] + self.height]

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
        return pos[0] - self.top_left[0], pos[1] - self.top_left[1]

    def get_at_display(self, pos):
        x = pos[0] - self.top_left[0]
        y = pos[1] - self.top_left[1]
        if not (0 <= x <= self.width):
            raise Exception("Invalid position")
        if not (0 <= y <= self.height):
            raise Exception("Invalid position")
        return self.get_at((x - 1, y - 1))

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

    def get_at_screen_position(self, x, y):
        # get colorvalue based on a screen position
        return self.data[y - self.top_left[1]][x - self.top_left[0]]

    #
    #   Setters
    #

    def set_top_left(self, pos):
        self.top_left = pos

    def set_at_display(self, pos, color):
        x = pos[0] - self.top_left[0] - 1
        y = pos[1] - self.top_left[1] - 1
        if not (0 <= x <= self.width - 1):
            raise Exception("Invalid position")
        if not (0 <= y <= self.height - 1):
            raise Exception("Invalid position")
        self.set_at((x, y), color)

    #
    #   Condition Statements
    #

    def in_display_image(self, pos):
        botton_right = self.get_botton_right()
        return self.top_left[0] <= pos[0] <= botton_right[0] and self.top_left[1] <= pos[1] <= botton_right[1]

    def set_at(self, pos, color):
        self.data[pos[1]][pos[0]] = color

    def set_at_band(self, pos, color, band):
        self.data[pos[1]][pos[0]][band] = color

    #
    #   Noises
    #

    def noise_gaussian(self, percent, mu=0, sigma=1):
        # Gaussian is Aditive
        function = ATIRandom.gaussian
        self.__additive_noise_two_parameters(percent, mu, sigma, function)
        return

    def __multiplicative_noise_one_parameter(self, percent, func_parameter, function):
        for x in range(self.width):
            for y in range(self.height):
                if ATIRandom.has_to_apply(percent):
                    variation = function(func_parameter)
                    new_color = [int(round(self.get_at((x, y))[0] * variation)),
                                 int(round(self.get_at((x, y))[1] * variation)),
                                 int(round(self.get_at((x, y))[2] * variation))]
                    new_color = ATIColor.fix_color(new_color)
                    self.set_at((x, y), new_color)
        return

    def __additive_noise_two_parameters(self, percent, func_parameter1, func_parameter2, function):
        for x in range(self.width):
            for y in range(self.height):
                if ATIRandom.has_to_apply(percent):
                    variation = function(func_parameter1, func_parameter2)
                    new_color = [int(round(self.get_at((x, y))[0] + variation)),
                                 int(round(self.get_at((x, y))[1] + variation)),
                                 int(round(self.get_at((x, y))[2] + variation))]
                    new_color = ATIColor.fix_color(new_color)
                    self.set_at((x, y), new_color)
        return

    def noise_rayleigh(self, percent, epsilon):
        # Rayleigh is Multiplicative
        function = ATIRandom.rayleigh
        self.__multiplicative_noise_one_parameter(percent, epsilon, function)

    def noise_exponential(self, percent, gamma):
        # Rayleigh is Multiplicative
        function = ATIRandom.exponential
        self.__multiplicative_noise_one_parameter(percent, gamma, function)

    def noise_salt_and_pepper(self, density):
        # Rayleigh is Multiplicative
        items_changed = 0
        total_items = 0
        for x in range(self.width):
            for y in range(self.height):
                total_items = total_items + 1
                new_random = ATIRandom.random()
                if new_random <= density:
                    items_changed = items_changed + 1
                    self.set_at((x, y), [0, 0, 0])
                if new_random >= (1 - density):
                    self.set_at((x, y), [self.max_gray_level, self.max_gray_level, self.max_gray_level])
        return

    #
    #   Image Operations
    #

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

        if self.__needs_compression():
            self.__scale_compression()
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
        self.__scale_compression()
        return

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

        self.__scale_compression()

    # Scalar product
    def scalar_product(self, scalar):
        if not isinstance(scalar, int):
            raise Exception('scalar is not an integer value')
        for x in range(self.width):
            for y in range(self.height):
                self.set_at((x, y), ATIColor.scalar_product(self.get_at((x, y)), scalar))
        self.dynamic_compression()

    # Threshold Function
    def __threshold_assign(self, array, threshold):
        # for x in range(3):
        max_value = self.max_gray_level
        min_value = 0
        if array[0] <= threshold:
            array[0] = min_value
            array[1] = min_value
            array[2] = min_value
        else:
            array[0] = max_value
            array[1] = max_value
            array[2] = max_value
        return array

    def threshold_function(self, threshold):
        if not isinstance(threshold, int):
            raise Exception('threshold is not an integer value')
        for x in range(self.width):
            for y in range(self.height):
                self.set_at((x, y), self.__threshold_assign(self.get_at((x, y)), threshold))

    def __scale_compression(self):
        self.__scale_compression_by_band(0)
        self.__scale_compression_by_band(1)
        self.__scale_compression_by_band(2)

    def fill_band_black(self, band):
        for x in range(self.width):
            for y in range(self.height):
                self.set_at_band((x, y), 0, band)

    def __scale_compression_by_band(self, band):
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
        color_deep = 256

        for x in range(self.width):
            for y in range(self.height):
                prev = self.get_at((x, y))[band]
                new = ((color_deep - 1) * math.log10(1 + prev)) / math.log10(1 + max_val)
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
        color_deep = 256
        for x in range(self.width):
            for y in range(self.height):
                prev = self.get_at((x, y))[band]
                new = math.pow((color_deep - 1), (1 - gamma)) * math.pow(prev, gamma)
                new = int(math.floor(new))
                self.set_at_band((x, y), new, band)

    def equalize_image(self):
        raise Exception("Not Implementd method")

    def color_count_array(self, band):
        array = [0] * 256
        for x in range(self.width):
            for y in range(self.height):
                array[self.get_at((x, y))[band]] = array[self.get_at((x, y))[band]] + 1
        return array

    def negative(self):
        for x in range(self.width):
            for y in range(self.height):
                color = self.get_at((x, y))
                self.set_at((x, y), (255 - color[0], 255 - color[1], 255 - color[2]))


def rgb_color_2_hsv_color(rgb_data):
    r = rgb_data[0]
    g = rgb_data[1]
    b = rgb_data[2]

    max_color = max(r, g, b)
    min_color = min(r, g, b)

    h, s, v = 0, 0, 0

    if max_color == min_color:
        h = "n/a"

    if max_color == r:
        if g >= b:
            h = round(60 * (g - b) / (max_color - min_color)) % 360
        else:
            h = (round(60 * (g - b) / (max_color - min_color)) + 360) % 360

    if max_color == g:
        h = (round(60 * (b - r) / (max_color - min_color)) + 120) % 360
    if max_color == b:
        h = (round(60 * (r - g) / (max_color - min_color)) + 240) % 360

    if max_color == 0:
        s = 0
    else:
        s = 1 - min_color / max_color

    v = max_color
    return [h, s, v]


def rgb2hsv(image_data, width, height):
    hsv_data = []
    for y in range(height):
        tmp_list = []
        for x in range(width):
            tmp_list.append(rgb_color_2_hsv_color(image_data[x][y]))
        hsv_data.append(tmp_list)
    return hsv_data


def hsv_color_2_rgb_color(hsv_data):
    h = hsv_data[0]
    s = hsv_data[1]
    v = hsv_data[2]

    if h > 360:
        h = math.fmod(h, 360)

    aux_h = math.ceil(h / 60) % 6
    f = ((math.ceil(h) / 60) % 6) - aux_h
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if aux_h == 0:
        return [v, t, p]
    if aux_h == 1:
        return [q, v, p]
    if aux_h == 2:
        return [p, v, t]
    if aux_h == 3:
        return [p, q, v]
    if aux_h == 4:
        return [t, p, v]
    if aux_h == 5:
        return [v, p, q]
    return "n/a"


def hsv2rgb(image_data, width, height):
    rgb_data = []
    for y in range(height):
        tmp_list = []
        for x in range(width):
            tmp_list.append(hsv_color_2_rgb_color(image_data[x][y]))
        rgb_data.append(tmp_list)
    return rgb_data


class ATIColor:
    def __init__(self):
        pass

    @classmethod
    def fix_color(cls, color_to_fix):
        if color_to_fix[0] < 0:
            color_to_fix[0] = 0
        if color_to_fix[1] < 0:
            color_to_fix[1] = 0
        if color_to_fix[2] < 0:
            color_to_fix[2] = 0
        if color_to_fix[0] > 255:
            color_to_fix[0] = 255
        if color_to_fix[1] > 255:
            color_to_fix[1] = 255
        if color_to_fix[2] > 255:
            color_to_fix[2] = 255
        return color_to_fix

    @classmethod
    def scalar_product(cls, color_array, scalar):
        for x in range(3):
            color_array[x] = color_array[x] * scalar
        return color_array
