from tkinter import filedialog, font, messagebox, ttk
import pygame
from pygame.locals import *
from ATIImage import *
from classes import *
import math
import numpy as np
import matplotlib.pyplot as plt


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.master.focus_set()

        self.menu = Menu(self.master)
        self.master.config(menu=self.menu)

        self.screenX = 0
        self.screenY = 0
        self.master.geometry("440x280")
        self.image_loaded = DISABLED

        self.file_menu = Menu(self.menu)
        self.file_submenu = Menu(self.file_menu)
        self.file_submenu.add_command(label="Circle", command=new_white_circle)
        self.file_submenu.add_command(label="Square", command=new_white_square)
        self.file_submenu.add_command(label="White empty image", command=open_new_image_window)
        self.file_submenu.add_command(label="Degrade", command=new_degrade_image)
        self.file_menu.add_cascade(label="New File", menu=self.file_submenu)

        self.file_menu.add_command(label="Load Image", command=open_file)
        self.file_menu.add_command(label="Save File", command=save_file)
        self.file_menu.add_command(label="Exit", command=self.exit_program)
        self.menu.add_cascade(label="File", menu=self.file_menu)

        self.edit_menu = Menu(self.menu)
        # self.edit_menu.add_command(label="Copy", command=self.copy_window)
        self.edit_menu.add_command(label="Reset Image", command=reset_image)
        self.edit_menu.add_command(label="Operations", command=operations_window)
        self.edit_menu.add_command(label="Threshold Image", command=threshold_window)
        self.edit_menu.add_command(label="Equalize Image", command=equalize_histogram)
        self.edit_menu.add_command(label="Negative", command=make_negative)
        self.edit_menu.add_command(label="Copy selection", command=copy_selection)
        self.edit_menu.add_command(label="Add Noise", command=open_noise_window)

        self.edit_submenu = Menu(self.edit_menu)
        self.edit_menu.add_cascade(label="Filter", menu=self.edit_submenu)
        self.edit_submenu.add_command(label="Average", command=lambda: set_kernel_size("avg"))
        self.edit_submenu.add_command(label="Median", command=lambda: set_kernel_size("mdn"))
        self.edit_submenu.add_command(label="Median weighted", command=filter_image_mdnp)
        self.edit_submenu.add_command(label="Gauss", command=lambda: set_kernel_size("gau"))
        self.edit_submenu.add_command(label="Border enhancement", command=set_edge_level)
        self.edit_submenu.add_command(label="Border detection (Sobel)", command=lambda: edge_enhance(1, "sobel"))
        self.edit_submenu.add_command(label="Border detection (Prewitt)", command=lambda: edge_enhance(1, "prewitt"))

        self.menu.add_cascade(label="Edit", menu=self.edit_menu)

        self.view_menu = Menu(self.menu)
        self.view_menu.add_command(label="Histogram", command=histogram_window)
        # self.view_menu.add_command(label="Equalize", command=equalize_histogram)
        self.menu.add_cascade(label="View", menu=self.view_menu)

        self.effect_menu = Menu(self.menu)
        self.effect_menu.add_command(label="Borders Detection", command=borders_window)
        self.effect_menu.add_command(label="Diffusion", command=diffusion_window)
        self.effect_menu.add_command(label="Thresholding Algoritms", command=thresholding_algoritm_window)
        self.menu.add_cascade(label="Effects", menu=self.effect_menu)

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

        self.hsv_h = StringVar()
        self.hsv_s = StringVar()
        self.hsv_v = StringVar()
        Label(master, text="H: ").grid(row=3, column=0)
        self.lblHValue = Label(master, text="0", textvariable=self.hsv_h)
        self.lblHValue.grid(row=3, column=1)
        Label(master, text=" S: ").grid(row=3, column=2)
        self.lblSValue = Label(master, text="0", textvariable=self.hsv_s)
        self.lblSValue.grid(row=3, column=3)
        Label(master, text=" V: ").grid(row=3, column=4)
        self.lblVValue = Label(master, text="0", textvariable=self.hsv_v)
        self.lblVValue.grid(row=3, column=5)

        Label(master, text="Selected region: ").grid(row=6, column=0)
        Label(master, text="Grey Average: ").grid(row=7, column=0)
        Label(master, text="Red Average: ").grid(row=8, column=0)
        Label(master, text="Green Average ").grid(row=9, column=0)
        Label(master, text="Blue Average ").grid(row=10, column=0)

        self.selection_pixel_count = Label(master, text="0")
        self.selection_pixel_count.grid(row=6, column=1)
        self.grey_pixel_average = Label(master, text="0")
        self.grey_pixel_average.grid(row=7, column=1)

        self.red_pixel_average = Label(master, text="0")
        self.red_pixel_average.grid(row=8, column=1)

        self.green_pixel_average = Label(master, text="0")
        self.green_pixel_average.grid(row=9, column=1)

        self.blue_pixel_average = Label(master, text="0")
        self.blue_pixel_average.grid(row=10, column=1)

    def set_focus(self):
        self.master.focus_set()

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
        hsv_color = ATIColor.rgb_to_hsv(value)
        self.xLabel['text'] = x
        self.yLabel['text'] = y
        self.valueEntry.delete(0, END)
        self.valueEntry.insert(0, value)
        self.hsv_h.set(hsv_color[0].__str__())
        self.hsv_s.set(hsv_color[1].__str__())
        self.hsv_v.set(hsv_color[2].__str__())

    def copy_window(self):
        self.__CopyWindow()

    class __CopyWindow:
        def __init__(self):
            self.window = Tk()
            self.window.focus_set()
            self.window.title("Copy Image")
            pass


#
#   Filters
#

def reset_image():
    editableImage.restore_data()
    draw_ati_image(editableImage)


def set_edge_level():
    window = Tk()
    window.focus_set()
    window.title("Edge enhancement level")
    Label(window, text="Level (0-1): ").grid(row=0, column=0)
    level = Entry(window)
    level.grid(row=0, column=1)
    Button(window, text="Change", command=lambda: edge_enhance(level.get(), "sobel", True)).grid(row=0, column=2)


def set_kernel_size(kernel_type):
    window = Tk()
    window.focus_set()
    window.title("Mask size")
    Label(window, text="Size: ").grid(row=0, column=0)
    size = Entry(window)
    size.grid(row=0, column=1)
    if kernel_type == "gau":
        Label(window, text="Sigma: ").grid(row=1, column=0)
        sigma = Entry(window)
        sigma.grid(row=1, column=1)
        Button(window, text="Change", command=lambda: which_filter(size.get(), kernel_type, sigma.get())) \
            .grid(row=0, column=2)
    else:
        Button(window, text="Change", command=lambda: which_filter(size.get(), kernel_type)).grid(row=0, column=2)


def which_filter(size, filter_type, sigma=None):
    if filter_type == "avg":
        filter_image_avg(size)
    elif filter_type == "mdn":
        filter_image_mdn(size)
    elif filter_type == "gau":
        filter_image_gauss(size, sigma)


"""def redraw_img(img, filtered_image):
    filtered_image = np.repeat(filtered_image, 3)
    filtered_image = filtered_image.reshape((img.shape[0], img.shape[1], 3))
    editableImage.data = filtered_image
    draw_ati_image(editableImage)"""


def redraw_img(filtered_image, col):
    img = np.array(editableImage.data)
    if not col:
        filtered_image = np.repeat(filtered_image, 3)
    filtered_image = filtered_image.reshape(img.shape)
    editableImage.data = filtered_image
    draw_ati_image(editableImage)


def filter_image_gauss(size, sigma):
    if editableImage.image_type == "ppm":
        colors = 3
    else:
        colors = 1
    image = np.array(editableImage.data)
    size = int(size)
    sigma = float(sigma)
    n_size = (size - 1) / 2
    pad = int((size - 1) / 2)
    x, y = np.mgrid[-n_size:n_size + 1, -n_size:n_size + 1]
    k = 1 / (2 * math.pi * sigma ** 2)
    gauss_kernel = k * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    mask = gauss_kernel / np.sum(gauss_kernel)
    fin_img = None
    for i in range(colors):
        img = image[:, :, i]
        gauss_img = convolve_func(img, mask, pad, size)
        if i < 1:
            fin_img = gauss_img
        else:
            fin_img = np.dstack((fin_img, gauss_img))
    if colors == 1:
        redraw_img(fin_img, False)
    else:
        redraw_img(fin_img, True)


def filter_image_mdnp():
    if editableImage.image_type == "ppm":
        colors = 3
    else:
        colors = 1
    image = np.array(editableImage.data)
    size = 3
    mask = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    pad = int((size - 1) / 2)
    fin_img = None
    for i in range(colors):
        img = image[:, :, i]
        mdn = convolve_func_mdnp(img, mask, pad, size)
        if i < 1:
            fin_img = mdn
        else:
            fin_img = np.dstack((fin_img, mdn))
    if colors == 1:
        redraw_img(fin_img, False)
    else:
        redraw_img(fin_img, True)


def filter_image_mdn(size):
    if editableImage.image_type == "ppm":
        colors = 3
    else:
        colors = 1
    image = np.array(editableImage.data)
    size = int(size)
    pad = int((size - 1) / 2)
    fin_img = None
    for i in range(colors):
        img = image[:, :, i]
        mdn = convolve_func_mdn(img, pad, size)
        if i < 1:
            fin_img = mdn
        else:
            fin_img = np.dstack((fin_img, mdn))
    if colors == 1:
        redraw_img(fin_img, False)
    else:
        redraw_img(fin_img, True)


def filter_image_avg(size):
    if editableImage.image_type == "ppm":
        colors = 3
    else:
        colors = 1
    image = np.array(editableImage.data)
    size = int(size)
    mask = np.ones((size, size))
    k = 1 / (size ** 2)
    mask = k * mask
    pad = int((size - 1) / 2)
    fin_img = None
    for i in range(colors):
        img = image[:, :, i]
        avg = convolve_func(img, mask, pad, size)
        if i < 1:
            fin_img = avg
        else:
            fin_img = np.dstack((fin_img, avg))
    if colors == 1:
        redraw_img(fin_img, False)
    else:
        redraw_img(fin_img, True)


def convolve_func_mdnp(img, mask, pad, size):
    output = np.zeros_like(img)
    image_padded = np.zeros((img.shape[0] + pad * 2, img.shape[1] + pad * 2))
    image_padded[pad:-pad, pad:-pad] = img
    flatmask = np.array(mask).flatten()
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            weightmdn = []
            flatarea = image_padded[y:y + size, x:x + size].flatten()
            for i in range(len(flatmask)):
                for j in range(flatmask[i]):
                    weightmdn.append(flatarea[i])
            output[y, x] = np.median(weightmdn)
    return output


def convolve_func_mdn(img, pad, size):
    output = np.zeros_like(img)
    image_padded = np.zeros((img.shape[0] + pad * 2, img.shape[1] + pad * 2))
    image_padded[pad:-pad, pad:-pad] = img
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            output[y, x] = np.median(image_padded[y:y + size, x:x + size])
    return output


def convolve_func(img, mask, pad, size):
    output = np.zeros_like(img)
    image_padded = np.zeros((img.shape[0] + pad * 2, img.shape[1] + pad * 2))
    image_padded[pad:-pad, pad:-pad] = img
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            output[y, x] = (mask * image_padded[y:y + size, x:x + size]).sum()
    return output


#
#   Noise
#

def open_noise_window():
    NoiseWindow()


class NoiseWindow:
    def __init__(self):
        self.window = Tk()
        self.window.focus_set()
        self.window.title("Add Noise")

        self.window.geometry("1020x220")
        #
        #    Title: Line 0
        #
        Label(self.window, text="Every operation is over the left image").grid(row=0, column=0)

        #
        #    Scale: Line 1
        #

        Label(self.window, text="Percent of the image: ").grid(row=1, column=0)
        self.sclPercent = Scale(self.window, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, length=200)
        self.sclPercent.grid(row=1, column=1)

        #
        #   Gaussian Additive Noise Line 2
        #   Label, Label Entry, Label Entry, Button
        #   Mu Entry: Float; Sigma Entry: Float
        #

        Label(self.window, text="Gaussian additive Noise; ").grid(row=2, column=0)
        mu_var = StringVar()
        sigma_var = StringVar()

        Label(self.window, text="Mu: ").grid(row=2, column=1)
        self.txtMu = Entry(self.window, textvariable=mu_var)
        self.txtMu.grid(row=2, column=2)

        Label(self.window, text="Sigma: ").grid(row=2, column=3)
        self.txtSigma = Entry(self.window, textvariable=sigma_var)
        self.txtSigma.grid(row=2, column=4)

        self.btnAddGaussian = Button(self.window, text="Add Gaussian noise",
                                     command=self.add_gaussian_noise)
        self.btnAddGaussian.grid(row=2, column=5)

        #
        #   Rayleigh Multiplicative Noise Line 3
        #   Label, Label Entry, Button
        #   Epsilon: float
        #

        Label(self.window, text="Rayleigh multiplicative Noise; ").grid(row=3, column=0)
        epsilon = StringVar()

        Label(self.window, text="Epsilon: ").grid(row=3, column=1)
        self.txtEpsilon = Entry(self.window, textvariable=epsilon)
        self.txtEpsilon.grid(row=3, column=2)

        self.btnAddRayleigh = Button(self.window, text="Add Rayleigh noise",
                                     command=self.add_rayleigh_noise)
        self.btnAddRayleigh.grid(row=3, column=5)

        #
        #   Exponential Multiplicative Noise Line 4
        #   Label, Label Entry, Button
        #   Gamma: float
        #

        Label(self.window, text="Exponential multiplicative Noise; ").grid(row=4, column=0)
        gamma = StringVar()

        Label(self.window, text="Gamma: ").grid(row=4, column=1)
        self.txtGamma = Entry(self.window, textvariable=gamma)
        self.txtGamma.grid(row=4, column=2)

        self.btnAddEpsilon = Button(self.window, text="Add Exponential noise",
                                    command=self.add_exponential_noise)
        self.btnAddEpsilon.grid(row=4, column=5)

        #
        #   Salt & Pepper Noise Line 5
        #   Label, Label Scale, Button
        #   Density: float [0, 0.5]
        #

        Label(self.window, text="Salt & Pepper Noise; ").grid(row=5, column=0)

        Label(self.window, text="Density: ").grid(row=5, column=1)
        self.sclDensity = Scale(self.window, from_=0, to=0.5, resolution=0.01, orient=HORIZONTAL)
        self.sclDensity.grid(row=5, column=2)

        self.btnAddSaltPepper = Button(self.window, text="Add Salt & Pepper noise",
                                       command=self.add_salt_pepper_noise)
        self.btnAddSaltPepper.grid(row=5, column=5)

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
        print(image.data[0])
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
        app.master.focus_set()
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
            editableImage.image_type = 'raw'
            RawWindow(file)
        if filename.lower().endswith('.pgm'):
            editableImage.image_type = 'pgm'
            load_pgm(file)
            draw_images()
        if filename.lower().endswith('.ppm'):
            editableImage.image_type = 'ppm'
            load_ppm(file)
            draw_images()
        file.close()
    else:
        print("cancelled")


class RawWindow:
    def __init__(self, file):

        self.window = Tk()
        self.window.focus_set()
        self.window.title("Load Raw file")
        self.window.geometry("280x140")
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
        app.master.focus_set()

        editableImage.height = height
        editableImage.width = width
        editableImage.data = image
        editableImage.max_gray_level = 255
        editableImage.set_restore_image()
        editableImage.values_set = True
        editableImage.image_type = ".raw"

        originalImage = editableImage.get_copy()
        originalImage.editable = False
        originalImage.id = 1
        originalImage.values_set = True
        originalImage.image_type = ".raw"

        draw_images()
        file.close()


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
    editableImage.set_restore_image()

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
    editableImage.set_restore_image()
    editableImage.values_set = True

    originalImage = editableImage.get_copy()
    originalImage.editable = False
    originalImage.id = 1
    originalImage.values_set = True
    originalImage.image_type = ".ppm"

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
            val = int(image.get_at((x, y))[0])
            file.write(int.to_bytes(val, length=1, byteorder="big"))
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
            val = int(image.get_at((x, y))[0])
            file.write(int.to_bytes(val, length=1, byteorder="big"))
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
            r_val = int(image.get_at((x, y))[0])
            g_val = int(image.get_at((x, y))[1])
            b_val = int(image.get_at((x, y))[2])
            file.write(int.to_bytes(r_val, length=1, byteorder="big"))
            file.write(int.to_bytes(g_val, length=1, byteorder="big"))
            file.write(int.to_bytes(b_val, length=1, byteorder="big"))
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

def operations_window():
    OperationsWindow()


class OperationsWindow:
    def __init__(self):
        self.window = Tk()
        self.window.focus_set()
        self.window.title("Operations")
        self.window.geometry("800x230")
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
        draw_ati_image(editableImage)


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


def threshold_window():
    ThresholdWindow()


class ThresholdWindow:
    def __init__(self):
        self.window = Tk()
        self.window.focus_set()
        self.window.title("Threshold Image")
        self.window.geometry("340x60")
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
        self.close_window()

    def close_window(self):
        self.window.destroy()
        app.focus_set()


#
#   Setters
#

def set_image(image, data, width, height, image_type, top_left, editable):
    image.data = data
    image.width = width
    image.height = height
    image.set_restore_image()
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


#
#   New Files
#
def open_new_image_window():
    NewImageWindow()


class NewImageWindow:
    def __init__(self):
        self.window = Tk()
        self.window.focus_set()
        self.window.title("New Image")

        self.window.geometry("380x160")

        self.font = font.Font(weight="bold")
        Label(self.window, text="Select image size", font=self.font).grid(row=0)

        Label(self.window, text="Width (px): ").grid(row=1, column=0)
        self.sclWidth = Scale(self.window, from_=1, to=720, resolution=16, orient=HORIZONTAL, length=250)
        self.sclWidth.grid(row=1, column=1)

        Label(self.window, text="Height (px): ").grid(row=2, column=0)
        self.sclHeight = Scale(self.window, from_=1, to=720, resolution=16, orient=HORIZONTAL, length=250)
        self.sclHeight.grid(row=2, column=1)

        self.btnCancel = Button(self.window, text="Cancel", command=self.close_window)
        self.btnCancel.grid(row=4, column=0)
        self.btnCreateFile = Button(self.window, text="Create Image", command=self.create_new_empty_image)
        self.btnCreateFile.grid(row=4, column=1)

    def close_window(self):
        self.window.destroy()
        app.master.focus_set()

    def create_new_empty_image(self):
        global editableImage, originalImage
        width = self.sclWidth.get()
        height = self.sclHeight.get()

        data = []
        for y in range(height):
            row = []
            for x in range(width):
                row.append([255, 255, 255])
            data.append(row)

        image = ATIImage()
        image.data = data
        image.width = width
        image.height = height
        image.set_restore_image()
        image.magic_num = 'P6'
        image.max_gray_level = 255
        editableImage = image
        originalImage = editableImage.get_copy()
        app.enable_image_menu()
        draw_images()
        self.close_window()


class NewWhiteCircle:
    def __init__(self):
        self.window = Tk()
        self.window.focus_set()
        self.window.title("New Circle")

        self.window.geometry("460x280")
        Label(self.window, text="Select size of image and circle").grid(row=0, column=0)

        Label(self.window, text="Width (px): ").grid(row=1, column=0)
        self.sclWidth = Scale(self.window, from_=1, to=200, resolution=8, orient=HORIZONTAL, length=250)
        self.sclWidth.grid(row=1, column=1)

        Label(self.window, text="Height (px): ").grid(row=2, column=0)
        self.sclHeight = Scale(self.window, from_=1, to=200, resolution=8, orient=HORIZONTAL, length=250)
        self.sclHeight.grid(row=2, column=1)

        Label(self.window, text="Radius (px): ").grid(row=3, column=0)
        self.sclRadius = Scale(self.window, from_=0, to=200,
                               orient=HORIZONTAL, length=250)
        self.sclRadius.grid(row=3, column=1)

        Label(self.window, text="Center x(px): ").grid(row=4, column=0)
        self.sclCenterX = Scale(self.window, from_=0, to=200, orient=HORIZONTAL, length=250)
        self.sclCenterX.grid(row=4, column=1)

        Label(self.window, text="Center y(px): ").grid(row=5, column=0)
        self.sclCenterY = Scale(self.window, from_=0, to=200, orient=HORIZONTAL, length=250)
        self.sclCenterY.grid(row=5, column=1)

        self.btnCancel = Button(self.window, text="Cancel", command=self.close_window)
        self.btnCancel.grid(row=6, column=0)
        self.btnCreateFile = Button(self.window, text="Create Image", command=self.create_new_circle_image)
        self.btnCreateFile.grid(row=6, column=1)

    def close_window(self):
        self.window.destroy()
        app.master.focus_set()

    def create_new_circle_image(self):
        global editableImage
        global originalImage

        data = []
        radius = self.sclRadius.get()
        center = [self.sclCenterX.get(), self.sclCenterY.get()]
        top_left = [20, 20]
        width = self.sclWidth.get()
        height = self.sclHeight.get()

        for y in range(height):
            row = []
            for x in range(width):
                if math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius:
                    row.append([0, 0, 0])
                else:
                    row.append([255, 255, 255])
            data.append(row)

        image = ATIImage(data=data, width=width, height=height, image_type='.ppm',
                         active=False, editable=True, top_left=top_left)
        image.max_gray_level = 255
        image.magic_num = 'P6'
        editableImage = image
        originalImage = image.get_copy()
        originalImage.set_top_left((image.top_left[0] + image.width, 20))
        draw_images()
        app.enable_image_menu()
        self.close_window()


def new_white_circle():
    NewWhiteCircle()


def new_white_square():
    NewWhiteSquare()


class NewWhiteSquare:
    def __init__(self):
        self.window = Tk()
        self.window.focus_set()
        self.window.title("New Square")

        self.window.geometry("460x280")
        Label(self.window, text="Select size of image and circle").grid(row=0, column=0)

        Label(self.window, text="Width (px): ").grid(row=1, column=0)
        self.sclWidth = Scale(self.window, from_=1, to=200, resolution=8, orient=HORIZONTAL, length=250)
        self.sclWidth.grid(row=1, column=1)

        Label(self.window, text="Height (px): ").grid(row=2, column=0)
        self.sclHeight = Scale(self.window, from_=1, to=200, resolution=8, orient=HORIZONTAL, length=250)
        self.sclHeight.grid(row=2, column=1)

        Label(self.window, text="Radius (px): ").grid(row=3, column=0)
        self.sclRadius = Scale(self.window, from_=0, to=200,
                               orient=HORIZONTAL, length=250)
        self.sclRadius.grid(row=3, column=1)

        Label(self.window, text="Center x(px): ").grid(row=4, column=0)
        self.sclCenterX = Scale(self.window, from_=0, to=200, orient=HORIZONTAL, length=250)
        self.sclCenterX.grid(row=4, column=1)

        Label(self.window, text="Center y(px): ").grid(row=5, column=0)
        self.sclCenterY = Scale(self.window, from_=0, to=200, orient=HORIZONTAL, length=250)
        self.sclCenterY.grid(row=5, column=1)

        self.btnCancel = Button(self.window, text="Cancel", command=self.close_window)
        self.btnCancel.grid(row=6, column=0)
        self.btnCreateFile = Button(self.window, text="Create Image", command=self.create_new_square_image)
        self.btnCreateFile.grid(row=6, column=1)

    def close_window(self):
        self.window.destroy()
        app.master.focus_set()

    def create_new_square_image(self):
        global editableImage
        global originalImage
        data = []
        height = self.sclHeight.get()
        width = self.sclWidth.get()
        top_left = [20, 20]
        radius = self.sclRadius.get()
        center = [self.sclCenterX.get(), self.sclCenterY.get()]

        for y in range(height):
            row = []
            for x in range(width):
                if abs(x - center[0]) <= radius and abs(y - center[1]) <= radius:
                    row.append((0, 0, 0))
                else:
                    row.append((255, 255, 255))
            data.append(row)

        image = ATIImage(data=data, width=width, height=height, image_type='.ppm', active=False,
                         editable=True, top_left=top_left)
        image.max_gray_level = 255
        image.magic_num = 'P6'
        editableImage = image
        originalImage = image.get_copy()
        originalImage.set_top_left((image.top_left[0] + image.width, 20))
        draw_images()
        app.enable_image_menu()
        self.close_window()


def new_degrade_image():
    NewDegrade()


class NewDegrade:
    def __init__(self):
        self.window = Tk()
        self.window.focus_set()
        self.window.title("New Degrade")

        Label(self.window, text="Select size of degrade image").grid(row=0, column=0)

        Label(self.window, text="Width (px): ").grid(row=1, column=0)
        self.sclWidth = Scale(self.window, from_=1, to=720, resolution=16, orient=HORIZONTAL, length=250)
        self.sclWidth.grid(row=1, column=1)

        Label(self.window, text="Height (px): ").grid(row=2, column=0)
        self.sclHeight = Scale(self.window, from_=1, to=720, resolution=16, orient=HORIZONTAL, length=250)
        self.sclHeight.grid(row=2, column=1)

        self.btnCancel = Button(self.window, text="Cancel", command=self.close_window)
        self.btnCancel.grid(row=6, column=0)

        self.btnCreateGreyDegrade = Button(self.window, text="Create Grey Degrade",
                                           command=self.create_new_grey_degrade)
        self.btnCreateGreyDegrade.grid(row=6, column=1)
        self.btnCreateColorDegrade = Button(self.window, text="Create Color Degrade",
                                            command=self.create_new_color_degrade)
        self.btnCreateColorDegrade.grid(row=6, column=2)

    def create_new_grey_degrade(self):
        function = ATIColor.grey_degrade
        self.create_new_degrade(function)
        pass

    def create_new_color_degrade(self):
        function = ATIColor.color_degrade
        self.create_new_degrade(function)
        pass

    def create_new_degrade(self, function):
        global editableImage
        global originalImage
        data = []
        height = self.sclHeight.get()
        width = self.sclWidth.get()
        top_left = [20, 20]
        for y in range(height):
            row = []
            for x in range(width):
                row.append(function(x, width))
            data.append(row)

        image = ATIImage(data=data, width=width, height=height, image_type='.ppm', active=False,
                         editable=True, top_left=top_left)
        image.max_gray_level = 255
        image.magic_num = 'P6'
        editableImage = image
        originalImage = image.get_copy()
        originalImage.set_top_left((image.top_left[0] + image.width, 20))
        draw_images()
        app.enable_image_menu()
        self.close_window()
        pass

    def close_window(self):
        self.window.destroy()


#
#   Draw
#


def draw_selection(x, y, x2, y2, selection_color):
    global surface
    top = min(y, y2)
    left = min(x, x2)
    right = max(x, x2)
    bottom = max(y, y2)
    for x in range(right - left):
        surface.set_at((x + left, top), selection_color)
        surface.set_at((x + left, bottom), selection_color)
    for y in range(bottom - top):
        surface.set_at((left, top + y), selection_color)
        surface.set_at((right, top + y), selection_color)


def draw_selection_rectangle(top_left, bottom_right, image_id):
    global surface
    top = top_left[1]
    left = top_left[0]
    bottom = bottom_right[1]
    right = bottom_right[0]
    image = get_image_by_id(image_id)
    for x in range(right - left):
        surface.set_at((x + left, top), image.get_at_display((x + left, top)))
        surface.set_at((x + left, bottom), image.get_at_display((x + left, bottom)))
    for y in range(bottom - top):
        surface.set_at((left, top + y), image.get_at_display((left, top + y)))
        surface.set_at((right, top + y), image.get_at_display((right, top + y)))


def draw_prev_selection_outside_img(top_left, bottom_right, col):
    top = top_left[1]
    left = top_left[0]
    bottom = bottom_right[1]
    right = bottom_right[0]
    for x in range(right - left):
        surface.set_at((x + left, top), col)
        surface.set_at((x + left, bottom), col)
    for y in range(bottom - top):
        surface.set_at((left, top + y), col)
        surface.set_at((right, top + y), col)


def draw_pre_image_selection(selection, image_id):
    tl, br = selection.get_image_within_selection()
    draw_selection_rectangle(tl, br, image_id)


def draw_ati_image(image):
    global surface
    height = image.height
    width = image.width
    pygame.display.get_surface()

    for x in range(0, width):
        for y in range(0, height):
            surface.set_at((x + image.top_left[0], y + image.top_left[1]), image.get_at([x, y]))


def draw_images():
    global editableImage
    global originalImage
    global surface
    editableImage.set_top_left((20, 20))
    originalImage.set_top_left((40 + originalImage.width, 20))
    editableImage.active = False
    originalImage.active = False
    pygame.display.set_mode((60 + editableImage.width * 2, 40 + editableImage.height))
    surface.fill((0, 0, 0))

    draw_ati_image(editableImage)
    draw_ati_image(originalImage)


#
#   View
#

def histogram_window():
    """Following libraries needed:
    import matplotlib.pyplot as plt
    import math"""
    y_vals, x_vals = editableImage.get_histogram(1, 0)  # or get editableimage in a more dynamic way
    plt.figure(figsize=[10, 8])
    plt.bar(x_vals, y_vals, width=5, color='#0504aa', alpha=0.7)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Frequency', fontsize=15)
    plt.title('Histogram', fontsize=15)
    plt.show()


def equalize_histogram():
    # y_values, x_values = get_histogram(editableImage.data, 1, 0)
    y_values, x_values = editableImage.get_histogram(1, 0)
    cs = cum_sum(y_values)
    cs = normalize(cs)
    image = np.asarray(editableImage.data)
    flat = image.flatten()
    image_new = cs[flat]
    image_new = np.reshape(image_new, image.shape)
    editableImage.data = image_new.astype("uint32")
    draw_ati_image(editableImage)


def cum_sum(hist):
    hist = iter(hist)
    cum_array = [next(hist)]
    for i in hist:
        cum_array.append(cum_array[-1] + i)
    return np.array(cum_array)


def normalize(cum_sum_pixel):
    n = (cum_sum_pixel - cum_sum_pixel.min())
    size = cum_sum_pixel.max() - cum_sum_pixel.min()
    cum_sum_pixel = (n / size) * 255
    cum_sum_pixel = np.rint(cum_sum_pixel)
    cum_sum_pixel = cum_sum_pixel.astype(int)
    return cum_sum_pixel


#
#   Effects
#

def borders_window():
    BordersWindow()


class BordersWindow:
    def __init__(self):
        self.window = Tk()
        self.window.focus_set()
        self.window.title("Effects")
        self.window.geometry("520x360")

        # -------------------------- Single operators --------------------------
        Label(self.window, text="Horizontal Prewitt").grid(row=1, column=0)
        self.btnHorizontalPrewitt = Button(self.window, text="Horizontal Prewitt",
                                           command=lambda: border_single_direction("prewitt"))
        self.btnHorizontalPrewitt.grid(row=1, column=1)

        Label(self.window, text="Vertical Prewitt").grid(row=1, column=2)
        self.btnVerticalPrewitt = Button(self.window, text="Vertical Prewitt",
                                         command=lambda: border_single_direction("prewitt", "vertical"))
        self.btnVerticalPrewitt.grid(row=1, column=3)

        Label(self.window, text="Horizontal Sobel").grid(row=2, column=0)
        self.btnHorizontalSobel = Button(self.window, text="Horizontal Sobel",
                                         command=lambda: border_single_direction("sobel"))
        self.btnHorizontalSobel.grid(row=2, column=1)

        Label(self.window, text="Vertical Sobel").grid(row=2, column=2)
        self.btnVerticalSobel = Button(self.window, text="Vertical Sobel",
                                       command=lambda: border_single_direction("sobel", "vertical"))
        self.btnVerticalSobel.grid(row=2, column=3)

        # -------------------------- Gradient Operators -----------------------------------

        Label(self.window, text="Prewitt").grid(row=3, column=0)
        self.btnHorizontalPrewitt = Button(self.window, text="Prewitt Abs", command=lambda: edge_enhance(1, "prewitt"))
        self.btnHorizontalPrewitt.grid(row=3, column=1)

        Label(self.window, text="Sobel").grid(row=4, column=0)
        self.btnHorizontalPrewitt = Button(self.window, text="Sobel Abs", command=lambda: edge_enhance(1, "sobel"))
        self.btnHorizontalPrewitt.grid(row=4, column=1)

        # ----------------------- Directional operators -----------------------
        self.angle = StringVar()
        posible_angles = [0, 45, 90, 135, 180, 225, 270, 315]

        Label(self.window, text="Directional Operators").grid(row=5, column=0)
        self.cbbAngle = ttk.Combobox(self.window, textvariable=self.angle, values=posible_angles)
        self.cbbAngle.grid(row=5, column=1)
        self.cbbAngle.current(0)

        Label(self.window, text="Prewitt").grid(row=6, column=0)
        self.btnDirectionalPrewitt = Button(self.window, text="Prewitt",
                                            command=lambda: self.directional_operator("prewitt"))
        self.btnDirectionalPrewitt.grid(row=6, column=1)

        Label(self.window, text="Sobel").grid(row=7, column=0)
        self.btnDirectionalSobel = Button(self.window, text="Sobel", command=lambda: self.directional_operator("sobel"))
        self.btnDirectionalSobel.grid(row=7, column=1)

        Label(self.window, text="Kirsh").grid(row=8, column=0)
        self.btnDirectionalKirsh = Button(self.window, text="Kirsh", command=lambda: self.directional_operator("kirsh"))
        self.btnDirectionalKirsh.grid(row=8, column=1)

        Label(self.window, text="Other").grid(row=9, column=0)
        self.btnDirectionalOther = Button(self.window, text="Other", command=lambda: self.directional_operator("other"))
        self.btnDirectionalOther.grid(row=9, column=1)

        # ------------------------------- Laplace methods -----------------------
        Label(self.window, text="Laplace method").grid(row=11, column=0)
        self.btnLaplaceMehtod = Button(self.window, text="Laplace", command=laplace_method)
        self.btnLaplaceMehtod.grid(row=11, column=1)

        Label(self.window, text="Laplace method with incline").grid(row=12, column=0)
        self.threshold = StringVar()
        self.txtThreshold = Entry(self.window, textvariable=self.threshold)
        self.txtThreshold.grid(row=12, column=1)
        self.btnLaplaceMethodIncline = Button(self.window, text="Incline",
                                              command=self.laplace_method_incline_wrapper)
        self.btnLaplaceMethodIncline.grid(row=12, column=3)

        Label(self.window, text="Laplace method Gausssian").grid(row=13, column=0)
        self.thresholdGaussian = StringVar()
        Label(self.window, text="Threshold").grid(row=13, column=1)
        self.txtThresholdGaussian = Entry(self.window, textvariable=self.thresholdGaussian)
        self.txtThresholdGaussian.grid(row=13, column=2)
        self.windowSize = StringVar()
        Label(self.window, text="sigma").grid(row=13, column=3)
        self.sigma = StringVar()
        self.txtSigma = Entry(self.window, textvariable=self.sigma)
        self.txtSigma.grid(row=13, column=4)

        Label(self.window, text="Window size").grid(row=13, column=5)
        self.txtWindowSize = Entry(self.window, textvariable=self.windowSize)
        self.txtWindowSize.grid(row=13, column=6)
        self.btnLaplaceGaussian = Button(self.window, text="Laplace Gaussian", command=self.laplace_gaussian)
        self.btnLaplaceGaussian.grid(row=13, column=7)

    def close_window(self):
        self.window.destroy()
        app.master.focus_set()

    def directional_operator(self, operator):
        angle = int(self.cbbAngle.get())
        edge_enhance(1, operator, angle)

    def laplace_method_incline_wrapper(self):
        threshold = int(self.txtThreshold.get())
        laplace_method_incline(threshold)

    def laplace_gaussian(self):
        threshold = int(self.txtThresholdGaussian.get())
        window_size = int(self.txtWindowSize.get())
        sigma = int(self.txtSigma.get())
        laplace_method_gaussian(threshold, window_size, sigma)


def border_single_direction(operator, side="horizontal"):
    if operator == "prewitt":
        mask = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    elif operator == "sobel":
        mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    if side == "vertical":
        mask = rotate_mask(mask, 90)

    print(mask)

    fin_img = None
    image = np.array(editableImage.data)
    size = 3
    pad = int((size - 1) / 2)
    colors = 3
    for i in range(colors):
        img = image[:, :, i]
        g_x = convolve_func(img, mask, pad, size)
        new_img = normalize(g_x)
        if i < 1:
            fin_img = new_img
        else:
            fin_img = np.dstack((fin_img, new_img))
    redraw_img(fin_img, True)


def apply_laplace_convolve():
    mask = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    fin_img = None
    image = np.array(editableImage.data)
    size = 3
    pad = int((size - 1) / 2)
    colors = 3
    for i in range(colors):
        img = image[:, :, i]
        g_x = convolve_func(img, mask, pad, size)
        if i < 1:
            fin_img = g_x
        else:
            fin_img = np.dstack((fin_img, g_x))
    return fin_img


def laplace_method():
    fin_img = apply_laplace_convolve()
    zero_cross_x = zero_cross(fin_img, editableImage.width, editableImage.height, "horizontal")
    zero_cross_y = zero_cross(fin_img, editableImage.width, editableImage.height, "vertical")

    editableImage.data = apply_synthesis_and(zero_cross_x, zero_cross_y, editableImage.width, editableImage.height)
    draw_ati_image(editableImage)


def laplace_method_incline(threshold):
    fin_img = apply_laplace_convolve()
    zero_cross_incline_x = calculate_derivative(fin_img, editableImage.width, editableImage.height, "horizontal",
                                                threshold)
    zero_cross_incline_y = calculate_derivative(fin_img, editableImage.width, editableImage.height, "vertical",
                                                threshold)
    editableImage.data = apply_synthesis_or(zero_cross_incline_x, zero_cross_incline_y, editableImage.width,
                                            editableImage.height)
    draw_ati_image(editableImage)


def laplace_method_gaussian(threshold, size, sigma):
    # Generar matriz
    # Convolucionar
    # Buscar pendiente de cruces por cero
    colors = 3
    image = np.array(editableImage.data)
    pad = int((size - 1) / 2)
    n_size = (size - 1) / 2
    x, y = np.mgrid[-n_size:n_size + 1, -n_size:n_size + 1]
    k = (-1 / (math.sqrt(2 * math.pi) * math.pow(sigma, 3)))
    mask = k * (2 - ((x ** 2 + y ** 2) / sigma ** 2)) * np.exp((-1) * (x ** 2 + y ** 2) / (2 * sigma ** 2))
    fin_img = None
    for i in range(colors):
        img = image[:, :, i]
        gauss_img = convolve_func(img, mask, pad, size)
        if i < 1:
            fin_img = gauss_img
        else:
            fin_img = np.dstack((fin_img, gauss_img))

    zero_cross_incline_x = calculate_derivative(fin_img, editableImage.width, editableImage.height, "horizontal",
                                                threshold)
    zero_cross_incline_y = calculate_derivative(fin_img, editableImage.width, editableImage.height, "vertical",
                                                threshold)
    editableImage.data = apply_synthesis_or(zero_cross_incline_x, zero_cross_incline_y, editableImage.width,
                                            editableImage.height)
    draw_ati_image(editableImage)


def apply_synthesis_and(matrix_x, matrix_y, width, height):
    new_matrix = []
    for y in range(height):
        row = []
        for x in range(width):
            color = []
            for z in range(3):
                color.append(min(matrix_x[y][x][z], matrix_y[y][x][z]))
            row.append(color)
        new_matrix.append(row)
    return new_matrix


def apply_synthesis_or(matrix_x, matrix_y, width, height):
    new_matrix = []
    for y in range(height):
        row = []
        for x in range(width):
            color = []
            for z in range(3):
                color.append(max(matrix_x[y][x][z], matrix_y[y][x][z]))
            row.append(color)
        new_matrix.append(row)
    return new_matrix


def rotate_mask(mask, angle):
    if angle == 0:
        return mask
    steps = angle / 45
    steps = steps % 8
    while steps > 0:
        rotate_matrix(mask)
        steps = steps - 1
    return mask


def rotate_matrix(mat):
    if not len(mat):
        return

        """ 
            top : starting row index 
            bottom : ending row index 
            left : starting column index 
            right : ending column index 
        """

    top = 0
    bottom = len(mat) - 1

    left = 0
    right = len(mat[0]) - 1

    while left < right and top < bottom:

        # Store the first element of next row,
        # this element will replace first element of
        # current row
        prev = mat[top + 1][left]

        # Move elements of top row one step right
        for i in range(left, right + 1):
            curr = mat[top][i]
            mat[top][i] = prev
            prev = curr

        top += 1

        # Move elements of rightmost column one step downwards
        for i in range(top, bottom + 1):
            curr = mat[i][right]
            mat[i][right] = prev
            prev = curr

        right -= 1

        # Move elements of bottom row one step left
        for i in range(right, left - 1, -1):
            curr = mat[bottom][i]
            mat[bottom][i] = prev
            prev = curr

        bottom -= 1

        # Move elements of leftmost column one step upwards
        for i in range(bottom, top - 1, -1):
            curr = mat[i][left]
            mat[i][left] = prev
            prev = curr

        left += 1

    return mat


def zero_cross(matrix, width, height, side):
    new_matrix = []
    if side == "horizontal":
        for y in range(height):
            row = []
            for x in range(width):
                color = []
                for z in range(3):
                    if x == (width - 1):
                        color.append(0)
                    elif (matrix[y][x][z] * matrix[y][x + 1][z]) < 0:
                        color.append(255)
                    elif x < (width - 2):
                        if matrix[y][x][z] * matrix[y][x + 2][z] < 0 and matrix[y][x + 1][z] == 0:
                            color.append(255)
                        else:
                            color.append(0)
                    else:
                        color.append(0)
                row.append(color)
            new_matrix.append(row)
    else:
        for x in range(width):
            row = []
            for y in range(height):
                color = []
                for z in range(3):
                    if y == (height - 1):
                        color.append(0)
                    elif (matrix[y][x][z] * matrix[y + 1][x][z]) < 0:
                        color.append(255)
                    elif y < height - 2:
                        if (matrix[y][x][z] * matrix[y + 2][x][z]) < 0 and matrix[y + 1][x][z] == 0:
                            color.append(255)
                        else:
                            color.append(0)
                    else:
                        color.append(0)
                row.append(color)
            new_matrix.append(row)
    return new_matrix


def apply_threshold(num1, num2, threshold):
    a = abs(num1)
    b = abs(num2)
    if (a + b) >= threshold:
        return 255
    return 0


def calculate_derivative(matrix, width, height, side, threshold):
    new_matrix = []
    if side == "horizontal":
        for y in range(height):
            row = []
            for x in range(width):
                pixel_color = []
                for z in range(3):
                    if x == width - 1:
                        pixel_color.append(0)
                    elif matrix[y][x][z] * matrix[y][x + 1][z] < 0:
                        pixel_color.append(apply_threshold(matrix[y][x][z], matrix[y][x + 1][z], threshold))
                    elif x < width - 2:
                        if matrix[y][x][z] * matrix[y][x + 2][z] < 0 and matrix[y][x + 1][z] == 0:
                            pixel_color.append(apply_threshold(matrix[y][x][z], matrix[y][x + 2][z], threshold))
                        else:
                            pixel_color.append(0)
                    else:
                        pixel_color.append(0)
                row.append(pixel_color)
            new_matrix.append(row)
    else:
        for x in range(width):
            row = []
            for y in range(height):
                pixel_color = []
                for z in range(3):
                    if y == height - 1:
                        pixel_color.append(0)
                    elif matrix[y][x][z] * matrix[y + 1][x][z] < 0:
                        pixel_color.append(apply_threshold(matrix[y][x][z], matrix[y + 1][x][z], threshold))
                    elif y < height - 2:
                        if matrix[y][x][z] * matrix[y + 2][x][z] < 0 and matrix[y + 1][x][z] == 0:
                            pixel_color.append(apply_threshold(matrix[y][x][z], matrix[y + 2][x][z], threshold))
                        else:
                            pixel_color.append(0)
                    else:
                        pixel_color.append(0)
                row.append(pixel_color)
            new_matrix.append(row)
    return new_matrix


def edge_enhance(level, operator, angle=0, enhance=False):
    level = float(level)
    if editableImage.image_type == "ppm":
        colors = 3
    else:
        colors = 1
    fin_img = None
    image = np.array(editableImage.data)
    if operator == "prewitt":
        h_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        h_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    elif operator == "sobel":
        h_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        h_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif operator == "kirsh":
        h_x = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
        h_y = np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])
    elif operator == "other":
        h_x = np.array([[1, 1, -1], [1, -2, -1], [1, 1, -1]])
        h_y = np.array([[1, 1, 1], [1, -2, 1], [-1, -1, -1]])
    elif operator == "laplace":
        h_x = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        h_y = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    h_x = rotate_mask(h_x, angle)
    h_y = rotate_mask(h_y, angle)
    size = 3
    pad = int((size - 1) / 2)
    for i in range(colors):
        img = image[:, :, i]
        g_x = convolve_func(img, h_x, pad, size)
        g_y = convolve_func(img, h_y, pad, size)
        g = np.sqrt(g_x ** 2 + g_y ** 2)
        if enhance:
            new_img = img + g * level
        """new_img = img + g * level"""
        new_img = normalize(new_img)
        if i < 1:
            fin_img = new_img
        else:
            fin_img = np.dstack((fin_img, new_img))
    if colors == 1:
        redraw_img(fin_img, False)
    else:
        redraw_img(fin_img, True)


#
# Diffusion
#

def diffusion_window():
    DifussionWindow()


class DifussionWindow:
    def __init__(self):
        self.window = Tk()
        self.window.focus_set()
        self.window.title("Diffusion")
        self.window.geometry("520x360")

        Label(self.window, text="times variable (T):").grid(row=0, column=0)
        self.times = StringVar()
        self.txtTimes = Entry(self.window, textvariable=self.times)
        self.txtTimes.grid(row=0, column=1)
        Label(self.window, text="Sigma").grid(row=0, column=2)
        self.sigma = StringVar()
        self.txtSigma = Entry(self.window, textvariable=self.sigma)
        self.txtSigma.grid(row=0, column=3)
        Label(self.window, text="Isotrópica").grid(row=1, column=0)
        self.btnDiffusionIsotropic = Button(self.window, text="Apply diffusion",
                                            command=self.isotropic_diffusion_wrapper)
        self.btnDiffusionIsotropic.grid(row=1, column=1)
        Label(self.window, text="Anisotropic").grid(row=2, column=0)
        self.btnDiffusionAnisotropic = Button(self.window, text="Apply diffusion",
                                              command=self.anisotropic_diffusion_wrapper)
        self.btnDiffusionAnisotropic.grid(row=2, column=1)

    def isotropic_diffusion_wrapper(self):
        times = int(self.txtTimes.get())
        isotropic_diffusion(times)

    def anisotropic_diffusion_wrapper(self):
        times = int(self.txtTimes.get())
        sigma = int(self.txtSigma.get())
        anisotropic_diffusion(times, sigma)


def isotropic_diffusion(times):
    new_matrix = editableImage.data
    width = editableImage.width
    height = editableImage.height
    for t in range(times):
        new_matrix = apply_isotropic(new_matrix, width, height)
    editableImage.data = new_matrix
    draw_ati_image(editableImage)
    return


def anisotropic_diffusion(times, sigma):
    new_matrix = editableImage.data
    width = editableImage.width
    height = editableImage.height
    for t in range(times):
        new_matrix = apply_anisotropic(new_matrix, width, height, sigma)

    editableImage.data = new_matrix
    draw_ati_image(editableImage)
    return


def apply_isotropic(matrix, width, height):
    new_matrix = []
    for y in range(height):
        row = []
        for x in range(width):
            pixel_color = []
            for z in range(3):
                dn = direction_difference(matrix, width, height, z, x, y, 1, 0)
                ds = direction_difference(matrix, width, height, z, x, y, -1, 0)
                de = direction_difference(matrix, width, height, z, x, y, 0, 1)
                do = direction_difference(matrix, width, height, z, x, y, 0, -1)
                pixel_color.append(matrix[y][x][z] + 0.25 * (dn + ds + de + do))
            row.append(pixel_color)
        new_matrix.append(row)
    return new_matrix


def apply_anisotropic(matrix, width, height, sigma):
    new_matrix = []
    for y in range(height):
        row = []
        for x in range(width):
            pixel_color = []
            for z in range(3):
                dn = direction_difference(matrix, width, height, z, x, y, 1, 0)
                ds = direction_difference(matrix, width, height, z, x, y, -1, 0)
                de = direction_difference(matrix, width, height, z, x, y, 0, 1)
                do = direction_difference(matrix, width, height, z, x, y, 0, -1)
                pixel_color.append(matrix[y][x][z] + 0.25 * (dn * leclerc_function(dn, sigma) +
                                                             ds * leclerc_function(ds, sigma) +
                                                             de * leclerc_function(de, sigma) +
                                                             do * leclerc_function(do, sigma)))
            row.append(pixel_color)
        new_matrix.append(row)
    return new_matrix


def direction_difference(matrix, width, height, range_color, x, y, difx, dify):
    new_x = x + difx
    new_y = y + dify

    value = 0
    if new_x < 0 or new_x >= width:
        value = 0
    elif new_y < 0 or new_y >= height:
        value = 0
    else:
        value = matrix[new_y][new_x][range_color]
    return value - matrix[y][x][range_color]


def leclerc_function(value, sigma):
    ans = np.exp((-1) * (value ** 2) / (sigma ** 2))
    return ans


#
# Global Thresholding
#

def thresholding_algoritm_window():
    ThresholdingAlgoritmWindow()


class ThresholdingAlgoritmWindow:
    def __init__(self):
        self.window = Tk()
        self.window.focus_set()
        self.window.title("Thresholding Algorithms")
        self.window.geometry("520x360")

        # ------------------------ Global Thresholding ----------------
        Label(self.window, text="Global Thresholding Algoritmn").grid(row=0, column=0)
        self.btnGlobalThresholding = Button(self.window, text="Global thresholding", command=self.global_thresholding)
        self.btnGlobalThresholding.grid(row=0, column=1)

        # ------------------------ Otsu algorithm ----------------------
        Label(self.window, text="Otsu algoritm").grid(row=1, column=0)
        self.btnOtsuThresholding = Button(self.window, text="Otsu thresholding", command=self.otsu_thresholding_wrapper)
        self.btnOtsuThresholding.grid(row=1, column=1)

        self.thresholdR = StringVar()
        self.thresholdG = StringVar()
        self.thresholdB = StringVar()

        Label(self.window, text="Calculated thresholds").grid(row=4, column=0)
        Label(self.window, text="R:").grid(row=5, column=0)
        self.lblRThreshold = Label(self.window, text="0", textvariable=self.thresholdR).grid(row=5, column=1)
        Label(self.window, text="G:").grid(row=6, column=0)
        self.lblGThreshold = Label(self.window, text="0", textvariable=self.thresholdG).grid(row=6, column=1)
        Label(self.window, text="B:").grid(row=7, column=0)
        self.lblBThreshold = Label(self.window, text="0", textvariable=self.thresholdB).grid(row=7, column=1)

        # self.txtGlobalThreshold = Entry(self.window, textvariable=self.global_thresholding)
        # self.txtGlobalThreshold.grid(row=0, column=2)

        # ----------------------- Otsu Thresholding ------------------
    def otsu_thresholding_wrapper(self):
        matrix = editableImage.data
        width = editableImage.width
        height = editableImage.height

        threshold = otsu_thresholding_algorithm(matrix, width, height)
        self.thresholdR.set(threshold[0])
        self.thresholdG.set(threshold[1])
        self.thresholdB.set(threshold[2])
        print(threshold)
        new_matrix = apply_thresholding_by_range(matrix, width, height, threshold)
        editableImage.data = new_matrix
        draw_ati_image(editableImage)


    def global_thresholding(self):
        matrix = editableImage.data
        width = editableImage.width
        height = editableImage.height

        threshold = global_thresholding_algorithm(matrix, width, height)
        self.thresholdR.set(threshold[0])
        self.thresholdB.set(threshold[1])
        self.thresholdG.set(threshold[2])
        print(threshold)

        new_matrix = apply_thresholding_by_range(matrix, width, height, threshold)
        editableImage.data = new_matrix
        draw_ati_image(editableImage)



def global_thresholding_algorithm(matrix, width, height):
    r = global_thresholding_by_range(matrix, width, height, 0)
    if editableImage.image_type == "ppm":
        g = global_thresholding_by_range(matrix, width, height, 1)
        b = global_thresholding_by_range(matrix, width, height, 2)
    else:
        g = r
        b = r
    threshold = [r, g, b]
    return threshold


def global_thresholding_by_range(matrix, width, height, color_range):
    min_value = get_min_value_matrix(matrix, width, height, color_range)
    max_value = get_max_value_matrix(matrix, width, height, color_range)
    if max_value == min_value or max_value - min_value == 1:
        return None
    t = ATIRandom.rand_int_between_range(min_value + 1, max_value - 1)
    new_t = global_thresholding_by_range_with_t(matrix, width, height, color_range, t)

    while abs(new_t - t) >= 1:
        t = new_t
        new_t = global_thresholding_by_range_with_t(matrix, width, height, color_range, t)
    return int(new_t)


def global_thresholding_by_range_with_t(matrix, width, height, color_range, t):
    m1 = get_global_thresholding_m1(matrix, width, height, color_range, t)
    m2 = get_global_thresholding_m2(matrix, width, height, color_range, t)
    return (m1 + m2) / 2


def get_global_thresholding_m1(matrix, width, height, color_range, t):
    n1 = 0
    amount = 0
    for y in range(height):
        for x in range(width):
            if matrix[y][x][color_range] <= t:
                amount = amount + matrix[y][x][color_range]
                n1 = n1 + 1
    return int(amount) / int(n1)


def get_global_thresholding_m2(matrix, width, height, color_range, t):
    n2 = 0
    amount = 0
    for y in range(height):
        for x in range(width):
            if matrix[y][x][color_range] > t:
                amount = amount + matrix[y][x][color_range]
                n2 = n2 + 1
    return int(amount) / int(n2)


def get_min_value_matrix(matrix, width, height, color_range):
    min_value = matrix[0][0][color_range]
    for y in range(height):
        for x in range(width):
            if (matrix[y][x][color_range] < min_value):
                min_value = matrix[y][x][color_range]
    return min_value


def get_max_value_matrix(matrix, width, height, color_range):
    max_value = matrix[0][0][color_range]
    for y in range(height):
        for x in range(width):
            if (matrix[y][x][color_range] > max_value):
                max_value = matrix[y][x][color_range]
    return max_value


def otsu_thresholding_algorithm(matrix, width, height):
    r = otsu_thresholding_by_range(matrix, width, height, 0)
    if editableImage.image_type == "ppm":
        g = otsu_thresholding_by_range(matrix, width, height, 1)
        b = otsu_thresholding_by_range(matrix, width, height, 2)
    else:
        g = r
        b = r
    threshold = [r, g, b]
    return threshold


def otsu_thresholding_by_range(matrix, width, height, color_range):
    histogram = make_histogram_by_range(matrix, width, height, color_range)
    acumulative_sums = count_acumulative_sums(histogram)
    acumulative_media = calculate_acumulative_media(histogram)
    media_global = acumulative_media[255]
    varianza = calculate_varianza(acumulative_sums, acumulative_media, media_global)
    indexes = get_max_value_index(varianza)
    if len(indexes) == 1:
        return indexes[0]
    count = 0
    for y in range(len(indexes)):
        count = count + indexes[y]
    return int(count/len(indexes))


def get_max_value_index(array):
    max_value = array[0]
    index_array = []
    for p in range(256):
        if array[p] > max_value:
            max_value = array[p]
            index_array = []
            index_array.append(p)
        elif array[p] == max_value:
            index_array.append(p)
    return index_array


def make_histogram_by_range(matrix, width, height, color_range):
    histogram = [0] * 256
    count = 0
    for y in range(height):
        for x in range(width):
            value = matrix[y][x][color_range]
            count = count + 1
            histogram[value] = histogram[value] + 1
    for t in range(256):
        histogram[t] = histogram[t] / count
    return histogram


def count_acumulative_sums(histogram):
    acumulative_sums = [0] * 256
    for p in range(256):
        acumulative_sums[p] = histogram[p]
        if p > 0:
            acumulative_sums[p] = acumulative_sums[p] + acumulative_sums[p - 1]
    return acumulative_sums


def calculate_acumulative_media(histogram):
    acumulative_media = [0] * 256
    for p in range(256):
        acumulative_media[p] = p * histogram[p]
        if p > 0:
            acumulative_media[p] = acumulative_media[p] + acumulative_media[p - 1]
    return acumulative_media


def calculate_varianza(acumulative_sums, acumulative_media, global_media):
    varianza = [0] * 256
    for p in range(256):
        if acumulative_sums[p] == 0 or acumulative_sums[p] == 1:
            varianza[p] = 0
        else:
            varianza[p] = (global_media * acumulative_sums[p] - acumulative_media[p])**2 / (acumulative_sums[p] * (1 - acumulative_sums[p]))
    return varianza


def apply_thresholding_by_range(matrix, width, height, thresholding):
    new_matrix = []
    for y in range(height):
        row = []
        for x in range(width):
            pixel_color = []
            for i in range(3):
                if matrix[y][x][i] > thresholding[i]:
                    pixel_color.append(255)
                else:
                    pixel_color.append(0)
            row.append(pixel_color)
        new_matrix.append(row)
    return new_matrix

#
#   Getters
#

def get_image_by_id(image_id):
    if image_id == 0:
        return editableImage
    elif image_id == 1:
        return originalImage
    raise Exception("Not valid image")


def is_click_in_images(pos):
    if editableImage.collidepoint(pos[0], pos[1]):
        return 0
    if originalImage.collidepoint(pos[0], pos[1]):
        return 1
    return -1


#
#   Selection
#

def update_selection_values(selection):
    image_id = -1
    for i in range(len(images)):
        image = get_image_by_id(i)
        if image.collidepoint(selection.new_x, selection.new_y):
            image_id = image.id
    if image_id != -1:
        image_selected = get_image_by_id(image_id)
        selected_data = image_data_in_selection(image_selected)
        app.selection_pixel_count["text"] = selection.get_pixel_count(selected_data)
        if image_selected.image_color_type() == 'g':
            app.grey_pixel_average["text"] = image_selected \
                .get_grey_average_display(selected_data)
        else:
            app.red_pixel_average["text"] = image_selected \
                .get_red_average_display(selected_data)
            app.green_pixel_average["text"] = image_selected \
                .get_green_average_display(selected_data)
            app.blue_pixel_average["text"] = image_selected \
                .get_blue_average_display(selected_data)
    return


def selection_on_image(selection, image):
    tl = selection.get_top_left()
    tr = selection.get_top_right()
    bl = selection.get_bottom_left()
    br = selection.get_bottom_right()
    tlcp = image.collidepoint(tl[0], tl[1])
    trcp = image.collidepoint(tr[0], tr[1])
    blcp = image.collidepoint(bl[0], bl[1])
    brcp = image.collidepoint(br[0], br[1])
    if tlcp or trcp or blcp or brcp:
        return True


def make_selection(selection):
    tl = selection.get_prev_top_left()
    br = selection.get_prev_bottom_right()
    draw_prev_selection_outside_img(tl, br, (0, 0, 0))
    for i in range(len(images)):
        image = get_image_by_id(i)
        if selection_on_image(selection, image):
            image.active = True
        if image.active:
            selection.image = image.id
            i_br = image.get_bottom_right()
            selection.set_image_within_selection(image.top_left, i_br, image.width, image.height)
            draw_pre_image_selection(selection, image.id)
    draw_selection(selection.x, selection.y, selection.new_x, selection.new_y, (0, 0, 255))


def image_data_in_selection(img):
    # function returning the color_data of image within current selection.
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
    done = False
    data = None
    original_in_selection = False
    while not done:
        for i in range(len(images)):
            img = get_image_by_id(i)
            if img.values_set:
                if img.editable:
                    if data:
                        for row in range(img.height):
                            for col in range(img.width):
                                img.set_at((col, row), (0, 0, 0))
                        draw_ati_image(img)
                        img.data = data
                        img.height = len(data)
                        img.width = len(data[0])
                        draw_ati_image(img)
                        done = True
                else:
                    data = image_data_in_selection(img)
                    if data:
                        original_in_selection = True
            else:
                print("Values not set")
                done = True
        if not original_in_selection:
            print("Copy original instead")
            done = True


def set_images_inactive():
    for i in range(len(images)):
        image = get_image_by_id(i)
        image.set_inactive()


#
#   Events Handlers
#

def handle_mouse_input(mouse_pos, image_click):
    if image_click != -1:
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
                mouse_position = pygame.mouse.get_pos()
                image_click = is_click_in_images(mouse_position)

                if is_selection_active:
                    draw_selection(new_selection.x, new_selection.y, new_selection.new_x, new_selection.new_y,
                                   (0, 0, 0))
                    orig = get_image_by_id(1)
                    edit = get_image_by_id(0)
                    draw_ati_image(orig)
                    draw_ati_image(edit)

                new_selection.set_start_pos(mouse_position)
                new_selection.set_image(image_click)

                update_selection_values(new_selection)

                dragging = True
                is_selection_active = True
                handle_mouse_input(mouse_position, image_click)

                last_action = "mousedown"
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                dragging = False
                set_images_inactive()
            last_action = "mouseup"
        elif event.type == MOUSEMOTION:
            if dragging:
                mouse_position = pygame.mouse.get_pos()
                new_selection.set_new_pos(mouse_position)
                make_selection(new_selection)
                update_selection_values(new_selection)
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
    """surface.fill((255, 255, 255))
    file = open("testing-images/Lenaclor.ppm", "rb")
    load_ppm(file)
    draw_images()
    file.close()"""

    done = False
    while not done:
        app.update()
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
editableImage.id = 0
originalImage = ATIImage()
originalImage.id = 1

images.append(editableImage)
images.append(originalImage)

if __name__ == '__main__':
    main()
