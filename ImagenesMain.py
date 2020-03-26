from tkinter import filedialog, font, messagebox
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
        self.file_menu.add_command(label="Save File", command=save_file, state=self.image_loaded)
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
        self.edit_submenu.add_command(label="Edge enhancement", command=set_edge_level)

        self.menu.add_cascade(label="Edit", menu=self.edit_menu)

        self.view_menu = Menu(self.menu)
        self.view_menu.add_command(label="Histogram", command=histogram_window)
        # self.view_menu.add_command(label="Equalize", command=equalize_histogram)
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

        Label(master, text="Pixel amount: ").grid(row=4, column=0)
        self.pixel_amount = Label(master, text="0")
        self.pixel_amount.grid(row=4, column=1)

        Label(master, text="Grayscale average: ").grid(row=5, column=0)
        self.gray_avg = Label(master, text="0")
        self.gray_avg.grid(row=5, column=1)

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
    imgdata = originalImage.data
    editableImage.data = imgdata
    draw_ati_image(editableImage)


def set_edge_level():
    window = Tk()
    window.focus_set()
    window.title("Edge enhancement level")
    Label(window, text="Level (0-1): ").grid(row=0, column=0)
    level = Entry(window)
    level.grid(row=0, column=1)
    Button(window, text="Change", command=lambda: edge_enhance(level.get())).grid(row=0, column=2)


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


def redraw_img(img, filtered_image):
    filtered_image = np.repeat(filtered_image, 3)
    filtered_image = filtered_image.reshape((img.shape[0], img.shape[1], 3))
    editableImage.data = filtered_image
    draw_ati_image(editableImage)


def edge_enhance(level):
    level = float(level)
    img = np.array(editableImage.data)[:, :, 0]
    h_x = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    h_y = np.matrix([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    size = 3
    pad = int((size - 1) / 2)
    g_x = convolve_func_avg(img, h_x, pad, size)
    g_y = convolve_func_avg(img, h_y, pad, size)
    g = np.sqrt(g_x ** 2 + g_y ** 2)
    const = level
    new_img = img + g * const
    new_img = normalize(new_img)
    redraw_img(img, new_img)


def filter_image_gauss(size, sigma):
    size = int(size)
    sigma = float(sigma)
    n_size = (size - 1) / 2
    pad = int((size - 1) / 2)
    img = np.array(editableImage.data)[:, :, 0]
    x, y = np.mgrid[-n_size:n_size + 1, -n_size:n_size + 1]
    k = 1 / (2 * math.pi * sigma ** 2)
    gauss_kernel = k * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    mask = gauss_kernel / np.sum(gauss_kernel)
    gauss_img = convolve_func_avg(img, mask, pad, size)
    redraw_img(img, gauss_img)


def filter_image_mdnp():
    size = 3
    mask = np.matrix([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    pad = int((size - 1) / 2)
    img = np.array(editableImage.data)[:, :, 0]
    mdn = convolve_func_mdnp(img, mask, pad, size)
    redraw_img(img, mdn)


def filter_image_mdn(size):
    size = int(size)
    pad = int((size - 1) / 2)
    img = np.array(editableImage.data)[:, :, 0]
    mdn = convolve_func_mdn(img, pad, size)
    redraw_img(img, mdn)


def filter_image_avg(size):
    size = int(size)
    mask = np.ones((size, size))
    k = 1 / (size ** 2)
    mask = k * mask
    pad = int((size - 1) / 2)
    img = np.array(editableImage.data)[:, :, 0]
    avg = convolve_func_avg(img, mask, pad, size)
    redraw_img(img, avg)


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


def convolve_func_avg(img, mask, pad, size):
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
        editableImage.max_gray_level = np.max(editableImage.data)

        originalImage = editableImage.get_copy()
        draw_images()
        file.close()

        # originalImage.data = image
        # originalImage.width = width
        # originalImage.height = height


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
        pass


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
                         active=True, editable=True, top_left=top_left)
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

        image = ATIImage(data=data, width=width, height=height, image_type='.ppm', active=True,
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

        image = ATIImage(data=data, width=width, height=height, image_type='.ppm', active=True,
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

    for x in range(0, width):
        for y in range(0, height):
            surface.set_at((x + image.top_left[0], y + image.top_left[1]), image.get_at([x, y]))


def draw_images():
    global editableImage
    global originalImage
    global surface
    editableImage.top_left = [20, 20]
    originalImage.top_left = [40 + originalImage.width, 20]
    editableImage.active = True
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
    y_vals, x_vals = get_histogram(editableImage.data, 1, 0)  # or get editableimage in a more dynamic way
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


def get_histogram(img_data, step, band):
    x_points = []
    y_points = []
    steps = int(round(255 / step))
    x_point = 0

    for i in range(steps + 1):
        y_points.append(0)
        x_points.append(x_point)
        x_point += step
    for row in img_data:
        for col in row:
            y_points[int(math.trunc(col[band] / step))] += 1
    return y_points, x_points


def equalize_histogram():
    y_values, x_values = get_histogram(editableImage.data, 1, 0)
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
#   Getters
#

def get_image_by_id(image_id):
    if image_id == 0:
        return editableImage
    if image_id == 1:
        return originalImage
    raise Exception("Not valid image")


def is_click_in_images(pos):
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
                # print("mousedown")
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
                mouse_position = pygame.mouse.get_pos()
                if is_click_in_images(mouse_position) == new_selection.image:
                    new_selection.set_new_pos(mouse_position)
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
