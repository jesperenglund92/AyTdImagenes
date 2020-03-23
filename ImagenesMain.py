from tkinter import filedialog, font, messagebox
import pygame
from pygame.locals import *
from ATIImage import *
from classes import *


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        self.menu = Menu(self.master)
        self.master.config(menu=self.menu)

        self.screenX = 0
        self.screenY = 0

        self.image_loaded = DISABLED

        self.file_menu = Menu(self.menu)
        self.file_submenu = Menu(self.file_menu)
        self.file_submenu.add_command(label="circle", command=new_white_circle)
        self.file_submenu.add_command(label="Square", command=new_white_square)
        self.file_submenu.add_command(label="Empty File", state=self.image_loaded)
        self.file_menu.add_cascade(label="New File", menu=self.file_submenu)

        self.file_menu.add_command(label="Load Image", command=open_file)
        self.file_menu.add_command(label="Save File", command=save_file, state=self.image_loaded)
        self.file_menu.add_command(label="Exit", command=self.exit_program)
        self.menu.add_cascade(label="File", menu=self.file_menu)

        self.edit_menu = Menu(self.menu)
        self.edit_menu.add_command(label="Copy", command=self.copy_window)
        self.edit_menu.add_command(label="Operations", command=self.operations_window)
        self.edit_menu.add_command(label="Threshold Image", command=self.threshold_window)
        self.edit_menu.add_command(label="Equalize Image", command=self.equalization_window)
        self.edit_menu.add_command(label="Negative", command=make_negative)
        self.edit_menu.add_command(label="Copy selection", command=copy_selection)
        self.edit_menu.add_command(label="Add Noise", command=self.open_noise_window)
        self.menu.add_cascade(label="Edit", menu=self.edit_menu)

        self.view_menu = Menu(self.menu)
        self.view_menu.add_command(label="HSV Color")
        self.view_menu.add_command(label="Histogram", command=self.histogram_window)
        self.menu.add_cascade(label="View", menu=self.view_menu)

        self.disable_image_menu()

        Label(master, text="x: ").grid(row=0, column=0)
        Label(master, text="y: ").grid(row=1, column=0)
        Label(master, text="color: ").grid(row=2, column=0)

        self.xLabel = Label(master, text="0")
        self.xLabel.grid(row=0, column=1)
        self.yLabel = Label(master, text="0")
        self.yLabel.grid(row=1, column=1)
        self.valueEntry = Entry(master, text="First Name")
        self.valueEntry.grid(row=2, column=1)
        self.btnChange = Button(master, text="Change",
                                command=lambda: change_pixel_val(self.xLabel['text'], self.yLabel['text'],
                                                                 self.valueEntry.get()))
        self.btnChange.grid(row=2, column=2)
        Label(master, text="Pixel amount: ").grid(row=3, column=0)
        self.pixel_amount = Label(master, text="0")
        self.pixel_amount.grid(row=3, column=1)
        Label(master, text="Grayscale average: ").grid(row=4, column=0)
        self.gray_avg = Label(master, text="0")
        self.gray_avg.grid(row=4, column=1)

        Label(master, text="Region seleccionada: ").grid(row=5, column=0)
        Label(master, text="Grey Average: ").grid(row=6, column=0)
        Label(master, text="Red Average: ").grid(row=7, column=0)
        Label(master, text="Green Average ").grid(row=8, column=0)
        Label(master, text="Blue Average ").grid(row=9, column=0)

        self.selection_pixel_count = Label(master, text="0")
        self.selection_pixel_count.grid(row=5, column=2)
        self.grey_pixel_average = Label(master, text="0")
        self.grey_pixel_average.grid(row=6, column=2)

        self.red_pixel_average = Label(master, text="0")
        self.red_pixel_average.grid(row=7, column=2)

        self.green_pixel_average = Label(master, text="0")
        self.green_pixel_average.grid(row=8, column=2)

        self.blue_pixel_average = Label(master, text="0")
        self.blue_pixel_average.grid(row=9, column=2)

    def exit_program(self):
        self.master.destroy()
        pygame.display.quit()
        pygame.quit()
        exit()
        pass

    def __emit_image_menu_action(self, state):

        self.edit_menu.entryconfigure(0, state=state)
        self.edit_menu.entryconfigure(1, state=state)
        self.edit_menu.entryconfigure(2, state=state)
        self.edit_menu.entryconfigure(3, state=state)
        self.edit_menu.entryconfigure(4, state=state)
        self.edit_menu.entryconfigure(5, state=state)
        self.edit_menu.entryconfigure(6, state=state)

        self.file_menu.entryconfigure(2, state=state)

        self.view_menu.entryconfigure(0, state=state)
        self.view_menu.entryconfigure(1, state=state)

    def disable_image_menu(self):
        self.__emit_image_menu_action(DISABLED)
        pass

    def enable_image_menu(self):
        self.__emit_image_menu_action(NORMAL)
        pass

    def set_value_entry(self, x, y, value):
        self.xLabel['text'] = x
        self.yLabel['text'] = y
        self.valueEntry.delete(0, END)
        self.valueEntry.insert(0, value)

    """def display_pixel_val(self, x, y, value, screen_x, screen_y):
        self.xLabel['text'] = x
        self.yLabel['text'] = y
        self.valueEntry.delete(0, END)
        self.valueEntry.insert(0, value)
        self.screenX = screen_x
        self.screenY = screen_y
    """

    def display_gray_pixel_amount(self, amount, gray_avg):
        self.pixel_amount['text'] = amount
        self.gray_avg['text'] = gray_avg

    def copy_window(self):
        self.__CopyWindow()

    class __CopyWindow:
        def __init__(self):
            self.window = Tk()
            self.window.focus_set()
            self.window.title("Copy Image")
            pass

    def operations_window(self):
        self.__OperationsWindow()

    class __OperationsWindow:
        def __init__(self):
            self.window = Tk()
            self.window.focus_set()
            self.window.title("Operations")

            Label(self.window, text="Every operation is over the left image and his original").grid(row=0, column=0)

            Label(self.window, text="Adding Images").grid(row=2, column=0)
            self.btnAddImage = Button(self.window, text="Add image with original",
                                      command=add_images)
            self.btnAddImage.grid(row=2, column=1)

            Label(self.window, text="Subtract Images").grid(row=3, column=0)
            self.btnSubtractImage = Button(self.window, text="Editable image subtract original",
                                           command=subtract_images)
            self.btnSubtractImage.grid(row=3, column=1)

            Label(self.window, text="Multiply Images").grid(row=4, column=0)
            self.btnMultiplyImage = Button(self.window, text="Editable image multiply original",
                                           command=multiply_image)
            self.btnMultiplyImage.grid(row=4, column=1)

            Label(self.window, text="Multiply Image by scalar").grid(row=6, column=0)
            # Here goes Entry for scalar
            scalar = StringVar()
            self.txtScalar = Entry(self.window, textvariable=scalar)
            self.txtScalar.grid(row=6, column=1)

            self.btnMultiplyImageByScalar = Button(self.window, text="Editable image multiply scalar",
                                                   command=self.multiply_images_scalar)
            self.btnMultiplyImageByScalar.grid(row=6, column=2)

            Label(self.window, text="Dynamic compression Images").grid(row=8, column=0)
            self.btnCompressDynamicRange = Button(self.window, text="Compress Image by Dynamic Range",
                                                  command=compression_dynamic_range)
            self.btnCompressDynamicRange.grid(row=8, column=1)

            Label(self.window, text="Gamma correction").grid(row=9, column=0)
            gamma = StringVar()
            self.txtGamma = Entry(self.window, textvariable=gamma)
            self.txtGamma.grid(row=9, column=1)
            self.btnGammaCorrection = Button(self.window, text="Apply Gamma correction",
                                             command=self.gamma_correction)
            self.btnGammaCorrection.grid(row=9, column=2)

            pass

        def multiply_images_scalar(self):
            scalar = int(self.txtScalar.get())
            editableImage.scalar_product(scalar)
            draw_ati_image(editableImage)
            return

        def gamma_correction(self):
            gamma = float(self.txtGamma.get())
            if gamma <= 0 or gamma == 1 or gamma >= 2:
                raise Exception("Invalid Gamma")
            editableImage.gamma_function(gamma)
            pass

    def threshold_window(self):
        self._ThresholdWindow()

    class _ThresholdWindow:
        def __init__(self):
            self.window = Tk()
            self.window.focus_set()
            self.window.title("Threshold Image")
            Label(self.window, text="Threshold: ").grid(row=0, column=0)

            self.threshold = StringVar()

            self.txtThreshold = Entry(self.window, textvariable=self.threshold)
            self.txtThreshold.grid(row=0, column=1)
            self.btnChange = Button(self.window, text="Change",
                                    command=self.apply_threshold_function)
            self.btnChange.grid(row=0, column=2)

        def apply_threshold_function(self):
            threshold = int(self.txtThreshold.get())
            # threshold = int(threshold)
            if threshold < 0 or threshold > 255:
                raise Exception("Only numbers between 0 and 255")

            editableImage.threshold_function(threshold)
            draw_ati_image(editableImage)

    def equalization_window(self):
        self.__EqualizationWindow()

    class __EqualizationWindow:
        def __init__(self):
            self.window = Tk()
            self.window.focus_set()
            self.window.title("Threshold Image")

    def histogram_window(self):
        """Following libraries needed:
        import matplotlib.pyplot as plt
        import math"""
        yvals, xvals = self.get_histogram(editableImage.data, 1, 0)  # or get editableimage in a more dynamic way
        plt.figure(figsize=[10, 8])
        plt.bar(xvals, yvals, width=5, color='#0504aa', alpha=0.7)
        # plt.xlim(0, max(xvals))
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylabel('Frequency', fontsize=15)
        plt.title('Histogram', fontsize=15)
        plt.show()
        # window = self.__HistogramWindow()

    def get_histogram(self, imgdata, step, band):
        xpoints = []
        ypoints = []
        steps = int(round(255 / step))
        xpoint = 0
        #for i in range(256):
        #    xpoints.append(i)
        #ypoints = editableImage.color_array(0)
        #return ypoints, xpoints

        for i in range(steps + 1):
            ypoints.append(0)
            xpoints.append(xpoint)
            xpoint += step
        for row in imgdata:
            for col in row:
                ypoints[int(math.trunc(col[band] / step))] += 1
        return ypoints, xpoints


    class __HistogramWindow():
        def __init__(self):
            self.window = Tk()
            self.window.focus_set()
            self.window.title("Histogram window")

    def open_noise_window(self):
        self.__NoiseWindow()

    class __NoiseWindow:
        def __init__(self):
            self.window = Tk()
            self.window.focus_set()
            self.window.title("Add Noise")

            #
            #    Title: Line 0
            #
            Label(self.window, text="Every operation is over the left image").grid(row=0, column=0)

            #
            #    Scale: Line 1 and 2
            #

            Label(self.window, text="Percent of the image: ").grid(row=1, column=0)
            self.sclPercent = Scale(self.window, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, length=200)
            self.sclPercent.grid(row=2)

            #
            #   Gaussian Additive Noise Line 3
            #   Label, Label Entry, Label Entry, Button
            #   Mu Entry: Float; Sigma Entry: Float
            #

            Label(self.window, text="Gaussian additive Noise; ").grid(row=3, column=0)
            mu_var = StringVar()
            sigma_var = StringVar()

            Label(self.window, text="Mu: ").grid(row=3, column=1)
            self.txtMu = Entry(self.window, textvariable=mu_var)
            self.txtMu.grid(row=3, column=2)

            Label(self.window, text="Sigma: ").grid(row=3, column=3)
            self.txtSigma = Entry(self.window, textvariable=sigma_var)
            self.txtSigma.grid(row=3, column=4)

            self.btnAddGaussian = Button(self.window, text="Add Gaussian noise",
                                         command=self.add_gaussian_noise)
            self.btnAddGaussian.grid(row=3, column=5)

            #
            #   Rayleigh Multiplicative Noise Line 4
            #   Label, Label Entry, Button
            #   Epsilon: float
            #

            Label(self.window, text="Rayleigh multiplicative Noise; ").grid(row=4, column=0)
            epsilon = StringVar()

            Label(self.window, text="Epsilon: ").grid(row=4, column=1)
            self.txtEpsilon = Entry(self.window, textvariable=epsilon)
            self.txtEpsilon.grid(row=4, column=2)

            self.btnAddRayleigh = Button(self.window, text="Add Rayleigh noise",
                                         command=self.add_rayleigh_noise)
            self.btnAddRayleigh.grid(row=4, column=5)

            #
            #   Exponential Multiplicative Noise Line 5
            #   Label, Label Entry, Button
            #   Gamma: float
            #

            Label(self.window, text="Exponential multiplicative Noise; ").grid(row=5, column=0)
            gamma = StringVar()

            Label(self.window, text="Gamma: ").grid(row=5, column=1)
            self.txtGamma = Entry(self.window, textvariable=gamma)
            self.txtGamma.grid(row=5, column=2)

            self.btnAddEpsilon = Button(self.window, text="Add Exponential noise",
                                        command=self.add_exponential_noise)
            self.btnAddEpsilon.grid(row=5, column=5)

            #
            #   Salt & Pepper Noise Line 6
            #   Label, Label Scale, Button
            #   Density: float [0, 0.5]
            #

            Label(self.window, text="Salt & Pepper Noise; ").grid(row=6, column=0)

            Label(self.window, text="Density: ").grid(row=6, column=1)
            self.sclDensity = Scale(self.window, from_=0, to=0.5, resolution=0.01, orient=HORIZONTAL)
            self.sclDensity.grid(row=6, column=2)

            self.btnAddSaltPepper = Button(self.window, text="Add Salt & Pepper noise",
                                           command=self.add_salt_pepper_noise)
            self.btnAddSaltPepper.grid(row=6, column=5)

        def add_gaussian_noise(self):
            percent = self.sclPercent.get()
            mu = self.txtMu.get()
            if mu == '':
                mu = float(0)
            sigma = self.txtSigma.get()
            if sigma == '':
                sigma = float(1)
            image_id = 0
            image = get_image_by_id(image_id)
            image.noise_gaussian(percent=percent, mu=mu, sigma=sigma)
            draw_ati_image(image)
            self.delete_window()

        def add_rayleigh_noise(self):
            percent = self.sclPercent.get()
            epsilon = self.txtEpsilon.get()
            image_id = 0
            image = get_image_by_id(image_id)
            if epsilon == '':
                raise Exception("Epsilon not set")
            image.noise_rayleigh(percent, epsilon)
            draw_ati_image(image)
            self.delete_window()

        def add_exponential_noise(self):
            percent = self.sclPercent.get()
            gamma = self.txtGamma.get()
            image_id = 0
            image = get_image_by_id(image_id)
            if gamma == '':
                raise Exception("Gamma not set")
            image.noise_exponential(percent, gamma)
            draw_ati_image(image)
            self.delete_window()

        def add_salt_pepper_noise(self):
            density = self.sclDensity.get()
            if density == '':
                raise Exception('Density is not set')
            density = float(density)
            if density > 0.5:
                raise Exception('Wrong Density value. Should be lower than 0.5')
            image_id = 0
            image = get_image_by_id(image_id)
            image.noise_salt_and_pepper(density)
            draw_ati_image(image)
            self.delete_window()

        def delete_window(self):
            self.window.destroy()
            del self


#
#   Open Files
#

def open_file():
    global editableImage
    global originalImage

    file_types = [
        ('RAW', '*.raw'),
        ('PGM', '*.pgm'),  # semicolon trick
        ('PPM', '*.ppm'),
        ('All files', '*'),
    ]
    filename = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=file_types)
    if filename:
        file = open(filename, "rb")
        if filename.lower().endswith('.raw'):
            editableImage.type = '.raw'
            RawWindow(file)
        if filename.lower().endswith('.pgm'):
            editableImage.type = '.pgm'
            load_pgm(file)
            draw_images()
        if filename.lower().endswith('.ppm'):
            editableImage.type = '.ppm'
            load_ppm(file)
            draw_images()
        file.close()
    else:
        print("cancelled")


class RawWindow:
    def __init__(self, file):

        self.window = Tk()
        self.window.focus_set()

        self.file = file
        self.font = font.Font(weight="bold")

        self.lblSelection = Label(self.window, text="Select Raw Size", font=self.font).grid(row=4)
        self.lblInitial = Label(self.window, text="Width").grid(row=5)
        self.lblFinal = Label(self.window, text="Height").grid(row=6)

        self.width = StringVar()
        self.height = StringVar()

        self.txtWidth = Entry(self.window, textvariable=self.width)
        self.txtHeight = Entry(self.window, textvariable=self.height)
        self.txtWidth.grid(row=5, column=1)
        self.txtHeight.grid(row=6, column=1)

        self.button = Button(self.window, text="Open raw", command=self.open_raw_image)
        self.button.grid(row=8)

    def open_raw_image(self):
        global editableImage
        global originalImage

        width = int(self.txtWidth.get())
        height = int(self.txtHeight.get())

        file = open(self.file.name, "rb")
        image = []

        for y in range(height):
            tmp_list = []
            for x in range(width):
                image_color = int.from_bytes(file.read(1), byteorder="big")
                tmp_list.append([image_color, image_color, image_color])
            image.append(tmp_list)

        app.enable_image_menu()
        self.window.destroy()

        editableImage.height = height
        editableImage.width = width
        editableImage.data = image

        originalImage = editableImage.get_copy()
        draw_images()
        file.close()

        #originalImage.data = image
        #originalImage.width = width
        #originalImage.height = height

def load_pgm(file):
    global editableImage
    global originalImage

    magic_num, width, height, max_val = read_ppm_pgm_header(file, '.pgm')
    image = []
    for y in range(height):
        tmp_list = []
        for x in range(width):
            pixel_color = int.from_bytes(file.read(1), byteorder="big")
            tmp_list.append([pixel_color, pixel_color, pixel_color])
        image.append(tmp_list)

    editableImage.data = image
    editableImage.width = width
    editableImage.height = height
    editableImage.magic_num = magic_num
    editableImage.max_gray_level = max_val

    originalImage = editableImage.get_copy()

    app.enable_image_menu()


def load_ppm(file):
    global editableImage
    global originalImage

    magic_num, width, height, max_val = read_ppm_pgm_header(file, '.ppm')
    image = []
    for y in range(height):
        tmp_list = []
        for x in range(width):
            tmp_list.append([int.from_bytes(file.read(1), byteorder="big"),
                             int.from_bytes(file.read(1), byteorder="big"),
                             int.from_bytes(file.read(1), byteorder="big")
                             ])
        image.append(tmp_list)

    editableImage.data = image
    editableImage.width = width
    editableImage.height = height
    editableImage.magic_num = magic_num
    editableImage.max_gray_level = max_val

    originalImage = editableImage.get_copy()

    app.enable_image_menu()


def read_ppm_pgm_header(file, image_type):
    count = 0
    magic_num = ''
    width = 0
    height = 0
    max_val = 255

    while count < 3:
        line = file.readline()
        if line[0] == '#':  # Ignore comments
            continue
        count = count + 1
        if count == 1:  # Magic num info
            magic_num = line.strip()
            magic_num = magic_num.decode('utf-8')
            if not (magic_num == 'P3' or magic_num == 'P6') and image_type == 'ppm':
                print('Not a valid PPM file')
            if not (magic_num == 'P2' or magic_num == 'P5') and image_type == 'pgm':
                print('Not a valid PGM file')
        elif count == 2:  # Width and Height
            [width, height] = (line.strip()).split()
            width = int(width)
            height = int(height)
        elif count == 3:  # Max gray level
            max_val = int(line.strip())
    return magic_num, width, height, max_val


#
#   Save Files
#

def save_file():
    file = filedialog.asksaveasfile(mode='w', defaultextension=editableImage.image_color_type())
    if file is None:
        return
    if file:
        if file.name.lower().endswith('.raw'):
            save_raw(file)
        if file.name.lower().endswith('.pgm'):
            save_pgm(file)
        if file.name.lower().endswith('.ppm'):
            save_ppm(file)
    pass


def save_raw(file):
    image = editableImage
    width = image.width
    height = image.height
    file.close()
    file = open(file.name, "wb")
    for y in range(height):
        for x in range(width):
            file.write(int.to_bytes(image.get_at((x, y))[0], length=1, byteorder="big"))
    file.close()
    messagebox.showinfo("File was successfully save", "The file is in: " + file.name)

    pass


def save_pgm(file):
    image = editableImage
    width = editableImage.width
    height = editableImage.height

    write_ppm_pgm_headers(file, image)
    file.close()
    file = open(file.name, 'ab')

    for y in range(height):
        for x in range(width):
            file.write(int.to_bytes(image.get_at((x, y))[0], length=1, byteorder="big"))
    file.close()

    messagebox.showinfo("File was successfully save", "The file is in: " + file.name)
    pass


def save_ppm(file):
    image = editableImage
    width = editableImage.width
    height = editableImage.height

    # Write Headers
    write_ppm_pgm_headers(file, image)

    file.close()
    file = open(file.name, "ab")

    for y in range(height):
        for x in range(width):
            file.write(int.to_bytes(image.get_at((x, y))[0], length=1, byteorder="big"))
            file.write(int.to_bytes(image.get_at((x, y))[1], length=1, byteorder="big"))
            file.write(int.to_bytes(image.get_at((x, y))[2], length=1, byteorder="big"))
    file.close()
    messagebox.showinfo("File was successfully save", "The file is in: " + file.name)
    pass


def write_ppm_pgm_headers(file, image):
    magic_num = image.magic_num
    if magic_num is None:
        if image.image_type == '.ppm':
            magic_num = 'P6'
        elif image.image_type == '.pgm':
            magic_num = 'P5'
        else:
            raise Exception("Invalid Image Type")

    file.write(magic_num)
    file.write('\n')
    file.write(image.width.__str__())
    file.write(' ')
    file.write(image.height.__str__())
    file.write('\n')
    max_gray_level = image.max_gray_level
    if max_gray_level is None:
        max_gray_level = 255
    file.write(max_gray_level.__str__())
    file.write('\n')


#
#   Operations with images
#

def add_images(image_id_1=0, image_id_2=1):
    image_1 = get_image_by_id(image_id_1)
    image_2 = get_image_by_id(image_id_2)
    image_1.add_image(image_2)
    draw_ati_image(image_1)


def subtract_images(image_id_1=0, image_id_2=1):
    image_1 = get_image_by_id(image_id_1)
    image_2 = get_image_by_id(image_id_2)
    image_1.subtract_image(image_2)
    draw_ati_image(image_1)


def multiply_image(image_id_1=0, image_id_2=1):
    image_1 = get_image_by_id(image_id_1)
    image_2 = get_image_by_id(image_id_2)
    image_1.multiply_image(image_2)
    draw_ati_image(image_1)


def compression_dynamic_range(image_id=0):
    image = get_image_by_id(image_id)
    image.dynamic_compression()
    draw_ati_image(image)


def make_negative(image_id=0):
    image = get_image_by_id(image_id)
    image.negative()
    draw_ati_image(image)


#
#   Setters
#

def set_image(image, data, width, height, image_type, top_left, editable):
    image.data = data
    image.width = width
    image.height = height
    image.image_type = image_type
    image.top_left = top_left
    image.editable = editable
    image.values_set = True


def change_pixel_val(x, y, pixel_color):
    global surface
    color_list = pixel_color.split()
    r, g, b = int(color_list[0]), int(color_list[1]), int(color_list[2])

    editableImage.data[y - 1][x - 1] = (r, g, b)
    edited_x = x + editableImage.get_top_left()[0]
    edited_y = y + editableImage.get_top_left()[1]
    pygame.display.get_surface()
    surface.set_at((edited_x, edited_y), (r, g, b))

    # for obj in objects:
    #    obj.data[x][y] = (r, g, b)
    #    draw_ati_image(obj)


#
#   Square & Circle
#

def new_white_circle():
    global editableImage
    global originalImage

    data = []
    radius = 50
    center = 100
    top_left = 50

    for i in range(200):
        row = []
        for j in range(200):
            if math.sqrt((i + top_left - center) ** 2 + (j + top_left - center) ** 2) <= radius:
                row.append((0, 0, 0))
            else:
                row.append((255, 255, 255))
        data.append(row)

    image = ATIImage(data=data, width=200, height=200, image_type='.ppm', active=True, editable=True, top_left=(20, 20))
    image.max_gray_level = 255
    image.magic_num = 'P6'
    editableImage = image
    originalImage = image.get_copy()
    originalImage.set_top_left((220, 20))
    draw_images()


def new_white_square():
    global editableImage
    global originalImage
    data = []
    height = 100
    width = 100
    tl_square = 50
    for i in range(200):
        row = []
        for j in range(200):
            if tl_square <= i <= tl_square + width and tl_square <= j <= tl_square + height:
                row.append((0, 0, 0))
            else:
                row.append((255, 255, 255))
        data.append(row)

    image = ATIImage(data=data, width=width, height=height, image_type='.ppm', active=True,
                     editable=True, top_left=(20, 20))
    image.max_gray_level = 255
    image.magic_num = 'P6'
    editableImage = image
    originalImage = image.get_copy()
    originalImage.set_top_left((220, 20))
    draw_images()


#
#   Draw
#

def draw_selection(x, y, x2, y2, selection_color):
    global surface
    top = min(y, y2)
    left = min(x, x2)
    right = max(x, x2)
    bottom = max(y, y2)
    pygame.display.get_surface()
    for x in range(right - left):
        surface.set_at((x + left, top), selection_color)
        surface.set_at((x + left, bottom), selection_color)
    for y in range(bottom - top):
        surface.set_at((left, top + y), selection_color)
        surface.set_at((right, top + y), selection_color)


def draw_selection_rectangle(selection, top_left, botton_rigth):
    global surface
    top = top_left[1]
    left = top_left[0]
    bottom = botton_rigth[1]
    right = botton_rigth[0]
    image = get_image_by_id(selection.image)
    pygame.display.get_surface()
    for x in range(right - left):
        surface.set_at((x + left, top), image.get_at_display((x + left, top)))
        surface.set_at((x + left, bottom), image.get_at_display((x + left, bottom)))
    for y in range(bottom - top):
        surface.set_at((left, top + y), image.get_at_display((left, top + y)))
        surface.set_at((right, top + y), image.get_at_display((right, top + y)))


def draw_pre_image_selection(selection):
    tl = selection.get_prev_top_left()
    br = selection.get_prev_botton_right()
    draw_selection_rectangle(selection, tl, br)


def draw_image_selection(selection):
    tl = selection.get_top_left()
    br = selection.get_botton_right()
    draw_selection_rectangle(selection, tl, br)


def draw_ati_image(image):
    global surface
    height = image.height
    width = image.width
    pygame.display.get_surface()
    for x in range(width):
        for y in range(height):
            surface.set_at((x + image.top_left[0], y + image.top_left[1]), image.get_at([x, y]))


def draw_images():
    global editableImage
    global originalImage
    global surface
    editableImage.top_left = [20, 20]
    originalImage.top_left = [40 + originalImage.width, 20]
    editableImage.active = True
    originalImage.active = False
    pygame.display.get_surface()
    surface.fill((0, 0, 0))

    pygame.display.set_mode((60 + editableImage.width * 2, 40 + editableImage.height))
    draw_ati_image(editableImage)
    draw_ati_image(originalImage)

    """
    f = filedialog.asksaveasfile(mode='w', defaultextension=".raw")
    if f:
        with open('blue_red_example.ppm', 'wb') as f:
            f.write(bytearray(ppm_header, 'ascii'))
            image.tofile(f)
    f.close()"""


#
#   Getters
#

def get_image_by_id(image_id):
    if image_id == 0:
        return editableImage
    if image_id == 1:
        return originalImage
    raise Exception("Not valid image")


def is_click_in_images(pos):
    # Tengo posiciones de Top Left y botton rigth. Puedo consultar con cualquier imagen
    if editableImage.in_display_image(pos):
        return 0
    if originalImage.in_display_image(pos):
        return 1
    return -1


#
#   Selection
#


def update_selection_values(selection):
    app.selection_pixel_count["text"] = selection.get_pixel_count()
    image_id = selection.image
    if image_id != -1:
        image_selected = get_image_by_id(image_id)
        if image_selected.image_color_type() == 'g':
            app.grey_pixel_average["text"] = image_selected \
                .get_grey_average_display(selection.get_top_left(), selection.get_botton_right())
        else:
            app.red_pixel_average["text"] = image_selected \
                .get_red_average_display(selection.get_top_left(), selection.get_botton_right())
            app.green_pixel_average["text"] = image_selected \
                .get_green_average_display(selection.get_top_left(), selection.get_botton_right())
            app.blue_pixel_average["text"] = image_selected \
                .get_blue_average_display(selection.get_top_left(), selection.get_botton_right())
    return


def make_selection(selection):
    draw_pre_image_selection(selection)
    draw_selection(selection.x, selection.y, selection.new_x, selection.new_y, (0, 0, 255))

    # rect = (x, y, x2-x, y2-y)
    # pygame.draw.rect(surface, (0,0,255), (x, y, x2-x, y2-y))


def image_data_in_selection(img):
    # function returning the color_data of in-image within current selection.
    data = []
    y_iterator = 0
    for i in range(abs(new_selection.new_y - new_selection.y)):
        row = []
        if new_selection.new_y < new_selection.y:
            y = new_selection.new_y
        else:
            y = new_selection.y
        x_iterator = 0
        for j in range(abs(new_selection.new_x - new_selection.x)):
            if new_selection.new_x < new_selection.x:
                x = new_selection.new_x
            else:
                x = new_selection.x
            if img.collidepoint(x + j, y + i):
                row.append(img.get_at_screen_position(j + x, y + i))
                x_iterator += 1

        if x_iterator > 0:
            y_iterator += 1
        if len(row) > 0:
            data.append(row)
    return data


def copy_selection():
    # function copying and drawing a copy of selected part of original image.
    for img in images:
        if img.values_set:
            if not img.editable:
                data = image_data_in_selection(img)
                image = ATIImage(data, len(data[0]), len(data), "type", (300, 50), True, True)
                images.append(image)
                draw_ati_image(image)


"""
def rgb_to_gray_scale(pixel_col):
    return pixel_col[0] * 0.3 + pixel_col[1] * 0.59 + pixel_col[2] * 0.11
"""

"""
def get_gray_pixel_amount(img):
    if img.values_set:
        data = img_data_in_selection(img)
        if len(data) > 0:
            width = len(data[0])
            height = len(data)
            pixel_amount = width * height
            count = 0
            for row in data:
                for col in row:
                    count += rgb_to_gray_scale(col)
            avg_gray = round(count / pixel_amount, 2)
            app.display_gray_pixel_amount(pixel_amount, avg_gray)
"""


#
#   Events Handlers
#

def handle_mouse_input(mouse_pos, image_click):
    image = get_image_by_id(image_click)
    pos_display = image.get_pos_display(mouse_pos)
    app.set_value_entry(pos_display[0], pos_display[1], image.get_at_display(mouse_pos))


def get_input():
    global dragging
    global start_x
    global start_y
    global new_selection
    global is_selection_active
    global last_action

    for event in pygame.event.get():
        if event.type == QUIT:
            return True
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:

                print("mousedown")

                mouse_position = pygame.mouse.get_pos()
                image_click = is_click_in_images(mouse_position)

                if editableImage.active and image_click != -1:
                    if is_selection_active:
                        draw_image_selection(new_selection)

                    new_selection.set_start_pos(mouse_position)
                    new_selection.set_image(image_click)

                    update_selection_values(new_selection)

                    dragging = True
                    is_selection_active = True

                    handle_mouse_input(mouse_position, image_click)

                last_action = "mousedown"
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                if last_action != "mousemotion":
                    is_selection_active = False
                    update_selection_values(new_selection)
                dragging = False
            last_action = "mouseup"
        elif event.type == MOUSEMOTION:
            if dragging:
                if is_click_in_images(pygame.mouse.get_pos()) == new_selection.image:
                    new_selection.set_new_pos(pygame.mouse.get_pos())
                    make_selection(new_selection)
                    update_selection_values(new_selection)
                else:
                    pass
            last_action = "mousemotion"
        sys.stdout.flush()  # get stuff to the console
    return False


#
#   General
#

def quit_callback():
    global Done
    Done = True


def main():
    # initialise pygame

    root.wm_title("Tkinter window")
    root.protocol("WM_DELETE_WINDOW", quit_callback)
    surface.fill((255, 255, 255))

    done = False
    while not done:
        try:
            app.update()
        except:
            print("dialog error")
        if get_input():
            done = True
        pygame.display.flip()


root = Tk()
pygame.init()
ScreenSize = (1, 1)
surface = pygame.display.set_mode(ScreenSize)
images = []
# list of images, in case we need to be more flexible than just one editable
# and one original image, possible to add more.
app = Window(root)
Done = False

dragging = False
start_x = None
start_y = None
new_selection = Selection()
is_selection_active = False
last_action = None

editableImage = ATIImage(editable=True)
originalImage = None

images.append(editableImage)
images.append(originalImage)

if __name__ == '__main__':
    main()
