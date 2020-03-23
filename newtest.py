from pygame.locals import *
from ATIImage import *
from classes import *
import pygame
import matplotlib.pyplot as plt
import math
import numpy as np

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        menu = Menu(self.master)
        self.master.config(menu=menu)

        self.screenX = 0
        self.screenY = 0


        file_menu = Menu(menu)
        file_submenu = Menu(file_menu)
        file_menu.add_cascade(label="New File", menu=file_submenu)

        file_menu.add_command(label="Exit", command=self.exitProgram)
        menu.add_cascade(label="File", menu=file_menu)

        edit_menu = Menu(menu)
        edit_submenu = Menu(edit_menu)
        edit_menu.add_cascade(label="Filter", menu=edit_submenu)
        edit_submenu.add_command(label="Average", command=lambda:self.set_kernel_size("avg"))
        edit_submenu.add_command(label="Median", command=lambda: self.set_kernel_size("mdn"))
        edit_submenu.add_command(label="Median weighted", command=filter_image_mdnp)
        edit_submenu.add_command(label="Gauss", command=lambda: self.set_kernel_size("gau"))
        edit_submenu.add_command(label="Edge enhancement", command=self.set_edge_level)
        menu.add_cascade(label="Edit", menu=edit_menu)

        view_menu = Menu(menu)
        view_menu.add_command(label="HSV Color")
        view_menu.add_command(label="Histogram", command=self.histogram_window)
        view_menu.add_command(label="Equalize", command=self.equalize_histogram)
        menu.add_cascade(label="View", menu = view_menu)

        Label(master, text="x: ").grid(row=0, column=0)
        Label(master, text="y: ").grid(row=1, column=0)
        Label(master, text="color: ").grid(row=2, column=0)

        self.xLabel = Label(master, text="0")
        self.xLabel.grid(row=0, column=1)
        self.yLabel = Label(master, text="0")
        self.yLabel.grid(row=1, column=1)
        self.valueEntry = Entry(master, text="First Name")
        self.valueEntry.grid(row=2, column=1)
        Label(master, text="Pixel amount: ").grid(row=3, column=0)
        self.pixel_amount = Label(master, text="0")
        self.pixel_amount.grid(row=3, column=1)
        Label(master, text="Grayscale average: ").grid(row=4, column=0)
        self.gray_avg = Label(master, text="0")
        self.gray_avg.grid(row=4, column=1)

        Label(master, text="Region seleccionada: ").grid(row=5, column=0)
        Label(master, text="Grey Average: ").grid(row=6, column=0)
        Label(master, text="Red Average: ").grid(row=7, column=0)
        Label(master, text="Green Average ").grid(row=8, column= 0)
        Label(master, text="Blue Average ").grid(row=9, column= 0)

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

    def exitProgram(self):
        pygame.display.quit()
        pygame.quit()
        exit()

    def setValueEntry(self, x, y, value):
        self.xLabel['text'] = x
        self.yLabel['text'] = y
        self.valueEntry.delete(0, END)
        self.valueEntry.insert(0, value)

    def display_pixelval(self, x, y, value, screenx, screeny):
        self.xLabel['text'] = x
        self.yLabel['text'] = y
        self.valueEntry.delete(0, END)
        self.valueEntry.insert(0, value)
        self.screenX = screenx
        self.screenY = screeny

    def display_gray_pixamount(self, amount, grayavg):
        self.pixel_amount['text'] = amount
        self.gray_avg['text'] = grayavg

    def histogram_window(self):
        """Following libraries needed:
        import matplotlib.pyplot as plt
        import math"""
        yvals, xvals = get_histogram(editableImage.data, 5, 0) # or get editableimage in a more dynamic way
        display_histogram(xvals, yvals)

    def equalize_histogram(self):
        yvals, xvals = get_histogram(editableImage.data, 1, 0)
        cs = cum_sum(yvals)
        cs = normalize(cs)
        img = np.asarray(editableImage.data)
        flat = img.flatten()
        img_new = cs[flat]
        img_new = np.reshape(img_new, img.shape)
        editableImage.data = img_new
        drawATIImage(editableImage)


    def set_edge_level(self):
        window = Tk()
        window.focus_set()
        window.title("Edge enhancement level")
        Label(window, text="Level (0-1): ").grid(row=0, column=0)
        level = Entry(window)
        level.grid(row=0, column=1)
        Button(window, text="Change", command=lambda: edge_enhance(level.get())).grid(row=0, column=2)


    def set_kernel_size(self, type):
        window = Tk()
        window.focus_set()
        window.title("Mask size")
        Label(window, text="Size: ").grid(row=0, column=0)
        size = Entry(window)
        size.grid(row=0, column=1)
        if type == "gau":
            Label(window, text="Sigma: ").grid(row=1, column=0)
            sigma = Entry(window)
            sigma.grid(row=1, column=1)
            Button(window, text="Change", command=lambda: which_filter(size.get(), type, sigma.get())).grid(row=0, column=2)
        else:
            Button(window, text="Change", command=lambda: which_filter(size.get(), type)).grid(row=0, column=2)


def which_filter(size, type, sigma=None):
    if type == "avg":
        filter_image_avg(size)
    elif type == "mdn":
        filter_image_mdn(size)
    elif type == "gau":
        filter_image_gauss(size, sigma)


def redraw_img(img, filtimg):
    filtimg = np.repeat(filtimg, 3)
    filtimg = filtimg.reshape((img.shape[0], img.shape[1], 3))
    editableImage.data = filtimg
    drawATIImage(editableImage)


def edge_enhance(level):
    level = float(level)
    img = np.array(originalImage.data)[:, :, 0]
    Hx = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Hy = np.matrix([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    size = 3
    pad = int((size - 1) / 2)
    Gx = convolve_func_avg(img, Hx, pad, size)
    Gy = convolve_func_avg(img, Hy, pad, size)
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    const = level
    new_img = img + G * const
    new_img = normalize(new_img)
    redraw_img(img, new_img)


def filter_image_gauss(size, sigma):
    size = int(size)
    sigma = float(sigma)
    N = (size - 1)/2
    pad = int((size - 1) / 2)
    img = np.array(originalImage.data)[:, :, 0]
    x, y = np.mgrid[-N:N + 1, -N:N + 1]
    k = 1/(2 * math.pi * sigma ** 2)
    gausskernel = k * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    mask = gausskernel/np.sum(gausskernel)
    gaussimg = convolve_func_avg(img, mask, pad, size)
    redraw_img(img, gaussimg)


def filter_image_mdnp():
    size = 3
    mask = np.matrix([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    pad = int((size - 1) / 2)
    img = np.array(originalImage.data)[:, :, 0]
    mdn = convolve_func_mdnp(img, mask, pad, size)
    redraw_img(img, mdn)


def filter_image_mdn(size):
    size = int(size)
    pad = int((size - 1) / 2)
    img = np.array(originalImage.data)[:, :, 0]
    mdn = convolve_func_mdn(img, pad, size)
    redraw_img(img, mdn)


def filter_image_avg(size):
    size = int(size)
    mask = np.ones((size, size))
    k = 1/(size**2)
    mask = k * mask
    pad = int((size - 1)/2)
    img = np.array(originalImage.data)[:, :, 0]
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
    image_padded = np.zeros((img.shape[0] + pad*2, img.shape[1] + pad*2))
    image_padded[pad:-pad, pad:-pad] = img
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            output[y, x] = (mask * image_padded[y:y + size, x:x + size]).sum()
    return output



#
#   Histogram
#

def display_histogram(xaxis, yaxis):
    plt.figure(figsize=[10, 8])
    plt.bar(xaxis, yaxis, width=5, color='#0504aa', alpha=0.7)
    #plt.xlim(0, max(xvals))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel('Frequency', fontsize=15)
    plt.title('Histogram', fontsize=15)
    plt.show()


def get_histogram(imgdata, step, band):
    xpoints = []
    ypoints = []
    steps = int(round(255/step))
    xpoint = 0
    for i in range(steps+1):
        ypoints.append(0)
        xpoints.append(xpoint)
        xpoint += step
    for row in imgdata:
        for col in row:
            ypoints[int(math.trunc(col[band]/step))] += 1
    return ypoints, xpoints


def cum_sum(hist):
    hist = iter(hist)
    cumarray = [next(hist)]
    for i in hist:
        cumarray.append(cumarray[-1] + i)
    return np.array(cumarray)


def normalize(cs):
    n = (cs - cs.min())
    N = cs.max() - cs.min()
    cs = (n / N) * 255
    cs = np.rint(cs)
    cs = cs.astype(int)
    return cs


#
#   Open
#

def open_raw():
    global editableImage
    global originalImage

    width = int(290)
    height = int(207)

    file = open("testing-images/barco.raw", "rb")
    image = []
    surface = pygame.display.set_mode((width, height))

    for y in range(height):
        tmpList = []
        for x in range(width):
            color = int.from_bytes(file.read(1), byteorder="big")

            surface.set_at((x, y), (color, color, color))
            tmpList.append([color, color, color])
        image.append(tmpList)

    editableImage.height = height
    editableImage.width = width
    editableImage.data = image

    originalImage.height = height
    originalImage.width = width
    originalImage.data = image
    drawImages()
    file.close()


def drawImages():
    global editableImage
    global originalImage
    editableImage.topleft = [20, 20]
    # originalImage.topleft = [40 + originalImage.width, 20]
    originalImage.set_top_left([40 + originalImage.width, 20])
    editableImage.active = True
    originalImage.active = False

    pygame.display.set_mode((60 + editableImage.width * 2, 40 + editableImage.height))
    drawATIImage(editableImage)
    drawATIImage(originalImage)


def drawATIImage(image):
    height = image.height
    width = image.width
    surface = pygame.display.get_surface()
    for x in range(width):
        for y in range(height):
            surface.set_at((x + image.topleft[0], y + image.topleft[1]), image.get_at([x, y]))


def quit_callback():
    global Done
    Done = True


def getInput():
    global dragging
    global startx
    global starty
    global newselection
    global isSelectionActive
    global lastaction

    for event in pygame.event.get():
        if event.type == QUIT:
            return True
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                lastaction = "mousedown"
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                dragging = False
            lastaction = "mouseup"
        elif event.type == MOUSEMOTION:
            lastaction="mousemotion"
        sys.stdout.flush()  # get stuff to the console
    return False


def main():
    # initialise pygame

    root.wm_title("Tkinter window")
    root.protocol("WM_DELETE_WINDOW", quit_callback)
    surface.fill((255, 255, 255))
    open_raw()
    done = False
    while not done:
        try:
            app.update()
        except:
            print("dialog error")
        if getInput():
            done = True
        pygame.display.flip()


root = Tk()
pygame.init()
ScreenSize = (1, 1)
surface = pygame.display.set_mode(ScreenSize)
images = [] #list of images, in case we need to be more flexible than just one editable and one original image, possible to add more.
app = Window(root)
Done = False

dragging = False
startx = None
starty = None
newselection = Selection()
isSelectionActive = False
lastaction = None

editableImage = ATIImage(editable=True)
originalImage = ATIImage()

images.append(editableImage)
images.append(originalImage)

if __name__ == '__main__': main()