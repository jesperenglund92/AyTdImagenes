import math

from tkinter import filedialog
from tkinter import *
import pygame
import sys
from pygame.locals import *
import struct
import binascii
import array
from threading import Thread
from queue import Queue


class PPM_Exception(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        menu = Menu(self.master)
        self.master.config(menu=menu)

        fileMenu = Menu(menu)

        newSubmenu = Menu(fileMenu)
        newSubmenu.add_command(label="circle", command=newWhiteCircle)

        fileMenu.add_cascade(label="New File", menu=newSubmenu)
        fileMenu.add_command(label="Load Image", command=openFile)
        fileMenu.add_command(label="Save File", command=saveFile)
        fileMenu.add_command(label="Exit", command=self.exitProgram)

        menu.add_cascade(label="File", menu=fileMenu)
        editMenu = Menu(menu)
        editMenu.add_command(label="Get value", command=getValue)
        editMenu.add_command(label="Edit value")
        menu.add_cascade(label="Edit", menu=editMenu)

        Label(master, text="x: ").grid(row=0, column=0)
        Label(master, text="y: ").grid(row=1, column=0)
        Label(master, text="color: ").grid(row=2, column=0)
        self.xLabel = Label(master, text="0")
        self.xLabel.grid(row=0, column=1)
        self.yLabel = Label(master, text="0")
        self.yLabel.grid(row=1, column=1)
        self.valueEntry = Entry(master, text="First Name")
        self.valueEntry.grid(row=2, column=1)

    def exitProgram(self):
        exit()

    def setValueEntry(self, x, y, value):
        self.xLabel['text'] = x
        self.yLabel['text'] = y
        self.valueEntry.delete(0,END)
        self.valueEntry.insert(0,value)


def loadPpm(file):
    count = 0
    while count < 3:
        line = file.readline()
        if line[0] == '#':  # Ignore comments
            continue
        count = count + 1
        if count == 1:  # Magic num info
            magicNum = line.strip()
            if magicNum != 'P2' and magicNum != 'P6':
                print('Not a valid PGM file')
        elif count == 2:  # Width and Height
            [width, height] = (line.strip()).split()
            width = int(width)
            height = int(height)
        elif count == 3:  # Max gray level
            maxVal = int(line.strip())
    image = []
    surface = pygame.display.set_mode((width, height))
    for y in range(height):
        tmpList = []
        for x in range(width):
            tmpList.append([int.from_bytes(file.read(1), byteorder="big"),
                            int.from_bytes(file.read(1), byteorder="big"),
                            int.from_bytes(file.read(1), byteorder="big")
                            ])
        image.append(tmpList)

    for y1 in range(0, height):
        for x1 in range(0, width):
            surface.set_at((x1, y1), image[y1][x1])
    pass


def loadPgm(file):
    pass


def loadRaw(file):
    pass


def getValue():
    pass

def saveFile():
    pass
    """
    f = filedialog.asksaveasfile(mode='w', defaultextension=".raw")
    if f:
        with open('blue_red_example.ppm', 'wb') as f:
            f.write(bytearray(ppm_header, 'ascii'))
            image.tofile(f)
    f.close()"""

def openFile():
    ftypes = [
        ('RAW', '*.raw'),
        ('PGM', '*.pgm'),  # semicolon trick
        ('PPM', '*.ppm'),
        ('All files', '*'),
    ]
    filename = filedialog.askopenfilename(initialdir="/", title="Select file", filetypes=ftypes)
    if filename:
        file = open(filename, "rb")
        if filename.lower().endswith(('.raw')):
            loadRaw(file)
        if filename.lower().endswith(('.pgm')):
            loadPgm(file)
        if filename.lower().endswith(('.ppm')):
            loadPpm(file)
        file.close()
    else:
        print("cancelled")


class Image:
    def __init__(self, data, width, height, type, surface, topleft=None):
        self.data = data
        self.width = width
        self.height = height
        self.type = type
        self.surface = surface
        self.topleft = topleft

    def draw(self):
        surface = pygame.display.get_surface()
        print(self.width)
        print(self.height)
        for x in range(self.height):
            for y in range(self.width):
                surface.set_at((x + self.topleft[0], y + self.topleft[0]), self.data[x][y])

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


class Selection:
    def __init__(self, posinitial, posfinal):
        self.posini = posinitial
        self.posfin = posfinal
        self.bottonrigth = [max(posinitial[0], posfinal[0]), max(posinitial[1], posfinal[1])]
        self.bottonleft = [min(posinitial[0], posfinal[0]), min(posinitial[1], posfinal[1])]

    def set_final(self, posfin):
        self.posfin = posfin
        self.updatePositions()

    def set_initial(self, posini):
        self.posini = posini
        self.updatePositions()

    def draw(self):
        pass

    def updatePositions(self):
        self.bottonrigth = [max(self.posini[0], self.posfin[0]), max(self.posini[1], self.posfin[1])]
        self.bottonleft = [min(self.posini[0], self.posfin[0]), min(self.posini[1], self.posfin[1])]


class square:
    def __init__(self, radius, pos):
        self.radius = radius
        self.pos = pos

    def belong2circle(self, point):
        return math.pow(point[0] - self.pos[0], 2) + math.pow(point[1] - self.pos[1], 2) <= math.pow(self.radius, 2)

# Transform RGB Method to HSV
def rgbcolor2hsvcolor(rgbdata):
    r = rgbdata[0]
    g = rgbdata[1]
    b = rgbdata[2]

    maxcolor = max(r, g, b)
    mincolor = min(r, g, b)

    s,v = 0, 0

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
        s = 1 - mincolor/maxcolor

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
    f = ((math.ceil(h)/60) % 6) - auxH
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


def checkOnImage(x, y):
    for obj in objects:
        if 50 <= x <= obj.width + 50 and 50 <= y <= obj.height + 50:
            return obj


def handleMouseinput(surface, app):
    x, y = pygame.mouse.get_pos()
    surface = pygame.display.get_surface()

    #clickedim = checkOnImage(x, y)
    #if clickedim:
    #    app.setValueEntry(x-50, y-50, clickedim.data[x-50][y-50])
    print(surface.get_at((x,y)))


def GetInput(surface, app):
    for event in pygame.event.get():
        if event.type == QUIT:
            return True
        print(event.type)
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                print("mouse 1")
                handleMouseinput(surface, app)
        sys.stdout.flush()  # get stuff to the console
    return False


Done = False


def quit_callback():
    global Done
    Done = True


def newBlackImage(width, height, surface):
    data = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append((0,0,0))
        data.append(row)
    return Image(data, width, height, "type", surface, (50, 50))


def newWhiteCircle():
    data = []
    radius = 50
    center = 150
    topleft = 50, 50
    for i in range(200):
        row = []
        for j in range(200):
            if math.sqrt((i + topleft[0] - center) ** 2 + (j + topleft[1] - center) ** 2) <= radius:
                row.append((255, 255, 255))
            else:
                row.append((0, 0, 0))
        data.append(row)
    image = Image(data, 200, 200, "type", pygame.display.get_surface(), topleft)
    image.draw()



def main():
    # initialise pygame
    """pygame.init()
    ScreenSize = (700, 400)
    surface = pygame.display.set_mode(ScreenSize)"""
    # initialise tkinter

    root.wm_title("Tkinter window")
    root.protocol("WM_DELETE_WINDOW", quit_callback)
    surface.fill((255, 255, 255))
    #blackImage = newBlackImage(200, 200, surface)
    #newWhiteCircle()
    #objects.append(blackImage)

    """thread = Thread(target=tkupdate)
    thread.daemon = True
    thread.start()"""
    # main loop
    while not Done:
        try:
            app.update()
        except:
            print("dialog error")
        if GetInput(surface, app):  # input event can also comes from diaglog
            break
        pygame.display.update()
    app.destroy()

root = Tk()
pygame.init()
ScreenSize = (700, 400)
surface = pygame.display.set_mode(ScreenSize)
objects = []
app = Window(root)

if __name__ == '__main__': main()
