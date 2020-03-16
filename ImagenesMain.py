import math

from tkinter import filedialog, font
from tkinter import *
import pygame
import sys
from pygame.locals import *
from ATIImage import *
from classes import *
import struct
import binascii
import array


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
        newSubmenu.add_command(label="Square", command=newWhiteSquare)

        fileMenu.add_cascade(label="New File", menu=newSubmenu)
        fileMenu.add_command(label="Load Image", command=openFile)
        fileMenu.add_command(label="Save File", command=saveFile)
        fileMenu.add_command(label="Exit", command=self.exitProgram)

        menu.add_cascade(label="File", menu=fileMenu)
        editMenu = Menu(menu)

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
        self.changebtn = Button(master, text="Change",
                                command=lambda: changepixval(self.xLabel['text'], self.yLabel['text'],
                                                             self.valueEntry.get()))
        self.changebtn.grid(row=2, column=2)

    def exitProgram(self):
        exit()

    def setValueEntry(self, x, y, value):
        self.xLabel['text'] = x
        self.yLabel['text'] = y
        self.valueEntry.delete(0, END)
        self.valueEntry.insert(0, value)


def loadPpm(file):
    count = 0
    while count < 3:
        line = file.readline()
        if line[0] == '#':  # Ignore comments
            continue
        count = count + 1
        if count == 1:  # Magic num info
            magicNum = line.strip()
            if magicNum != 'P2' or magicNum != 'P5':
                print('Not a valid PPM file')
        elif count == 2:  # Width and Height
            [width, height] = (line.strip()).split()
            width = int(width)
            height = int(height)
        elif count == 3:  # Max gray level
            maxVal = int(line.strip())
    image = []
    # surface = pygame.display.set_mode((width, height))
    for y in range(height):
        tmpList = []
        for x in range(width):
            tmpList.append([int.from_bytes(file.read(1), byteorder="big"),
                            int.from_bytes(file.read(1), byteorder="big"),
                            int.from_bytes(file.read(1), byteorder="big")
                            ])
        image.append(tmpList)

    editableImage.data = image
    editableImage.width = width
    editableImage.height = height

    originalImage.data = image
    originalImage.width = width
    originalImage.height = height


def savePpm(file):
    image = editableImage.get_data()
    width = editableImage.get_size()[0]
    height = editableImage.get_size()[1]
    ## TODO: Write headers

    for y in range(height):
        for x in range(width):
            file.write(int.to_bytes(image[x][y][0], byteorder="big"))
            file.write(int.to_bytes(image[x][y][1], byteorder="big"))
            file.write(int.to_bytes(image[x][y][2], byteorder="big"))
    pass


def loadPgm(file):
    count = 0
    while count < 3:
        line = file.readline()
        if line[0] == '#':  # Ignore comments
            continue
        count = count + 1
        if count == 1:  # Magic num info
            magicNum = line.strip()
            if magicNum != 'P2' or magicNum != 'P6':
                print('Not a valid PPM file')
        elif count == 2:  # Width and Height
            [width, height] = (line.strip()).split()
            width = int(width)
            height = int(height)
        elif count == 3:  # Max gray level
            maxVal = int(line.strip())
    image = []
    for y in range(height):
        tmpList = []
        for x in range(width):
            color = int.from_bytes(file.read(1), byteorder="big")
            tmpList.append([color, color, color])
        image.append(tmpList)

    editableImage.data = image
    editableImage.width = width
    editableImage.height = height

    originalImage.data = image
    originalImage.width = width
    originalImage.height = height


def savePgm(file):
    image = editableImage.get_data()
    width = editableImage.get_size()[0]
    height = editableImage.get_size()[1]
    ## TODO: Write headers

    for y in range(height):
        for x in range(width):
            file.write(int.to_bytes(image[x][y], byteorder="big"))
    pass


class RawWindow:
    def __init__(self, file):

        self.window = Tk()
        self.window.focus_set()

        self.file = file
        fuente = font.Font(weight="bold")

        self.lblSelection = Label(self.window, text="Select Raw Size", font=fuente).grid(row=4)
        self.lblInitial = Label(self.window, text="Width").grid(row=5)
        self.lblFinal = Label(self.window, text="Height").grid(row=6)

        self.width = StringVar()
        self.height = StringVar()

        self.txtWidth = Entry(self.window, textvariable=self.width)
        self.txtHeight = Entry(self.window, textvariable=self.height)
        self.txtWidth.grid(row=5, column=1)
        self.txtHeight.grid(row=6, column=1)

        self.button = Button(self.window, text="Open raw", command=self.openRawImage)
        self.button.grid(row=8)

    def openRawImage(self):
        print("Width: " + self.txtWidth.get() + " ; Height: " + self.txtHeight.get())

        width = int(self.txtWidth.get())
        height = int(self.txtHeight.get())

        file = open(self.file.name, "rb")
        image = []
        surface = pygame.display.set_mode((width, height))

        for y in range(height):
            tmpList = []
            for x in range(width):
                color = int.from_bytes(file.read(1), byteorder="big")

                surface.set_at((x, y), (color, color, color))
                tmpList.append([color, color, color])
            image.append(tmpList)
        self.window.destroy()

        editableImage.height = height
        editableImage.width = width
        editableImage.data = image

        originalImage.height = height
        originalImage.width = width
        originalImage.data = image
        printImages()
        file.close()


def loadRaw(file):
    window = RawWindow(file)


def saveRaw(file):
    image = editableImage.get_data()
    width = editableImage.get_size()[0]
    height = editableImage.get_size()[1]

    # surface = pygame.display.set_mode((width, height))
    for y in range(height):
        for x in range(width):
            color = int.to_bytes(image[x][y], byteorder="big")
            file.write(color)
    pass


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
            editableImage.type = '.raw'
            originalImage.type = '.raw'
            loadRaw(file)
        if filename.lower().endswith(('.pgm')):
            editableImage.type = '.pgm'
            editableImage.type = '.pgm'
            loadPgm(file)
            printImages()

        if filename.lower().endswith(('.ppm')):
            editableImage.type = '.ppm'
            editableImage.type = '.ppm'
            loadPpm(file)
            printImages()
        file.close()

    else:
        print("cancelled")


def printImages():
    editableImage.topleft = [20, 20]
    originalImage.topleft = [20 + editableImage.width + 20, 20]
    pygame.display.set_mode((60 + editableImage.width * 2, 40 + editableImage.height))


    drawATIImage(editableImage)
    drawATIImage(originalImage)


def saveFile():
    file = filedialog.asksaveasfile(mode='w', defaultextension=editableImage.get_type())
    if file:
        if file.name.lower().endswith(('.raw')):
            saveRaw(file)
        if file.name.lower().endswith(('.pgm')):
            savePgm(file)
        if file.name.lower().endswith(('.ppm')):
            savePpm(file)
    pass
    """
    f = filedialog.asksaveasfile(mode='w', defaultextension=".raw")
    if f:
        with open('blue_red_example.ppm', 'wb') as f:
            f.write(bytearray(ppm_header, 'ascii'))
            image.tofile(f)
    f.close()"""


def drawATIImage(image):
    height = image.height
    width = image.width
    surface = pygame.display.get_surface()
    for x in range(width):
        for y in range(height):
            surface.set_at((x + image.topleft[0], y + image.topleft[1]), image.get_at([x, y]))


def openRAWWindow():
    rawWindow = Tk()
    rawWindow.title("Select width and heigth")
    rawWindow.focus_set()
    lblSelection = Label(rawWindow, text="width").grid(row=1)
    lblInitial = Label(rawWindow, text="height").grid(row=3)


def quit_callback():
    global Done
    Done = True


def changepixval(x, y, color):
    colorlist = color.split()
    r, g, b = int(colorlist[0]), int(colorlist[1]), int(colorlist[2])
    for obj in objects:
        obj.data[x][y] = (r, g, b)
        drawATIImage(obj)


def newWhiteCircle():
    data = []
    radius = 50
    center = 150
    topleft = 50
    for i in range(200):
        row = []
        for j in range(200):
            if math.sqrt((i + topleft - center) ** 2 + (j + topleft - center) ** 2) <= radius:
                row.append((255, 255, 255))
            else:
                row.append((0, 0, 0))
        data.append(row)
    image = ATIImage(data, 200, 200, "type", (topleft, topleft))
    objects.append(image)
    drawATIImage(image)


def newWhiteSquare():
    data = []
    height = 100
    width = 100
    topleft = 50
    tlsquare = 50
    for i in range(200):
        row = []
        for j in range(200):
            if tlsquare <= i <= tlsquare + width and tlsquare <= j <= tlsquare + height:
                row.append((255, 255, 255))
            else:
                row.append((0, 0, 0))
        data.append(row)
    editableImage.data = data
    editableImage.width = 200
    editableImage.height = 200
    editableImage.type = "type"
    editableImage.topleft = (topleft, topleft)
    drawATIImage(editableImage)
    """image = ATIImage(data, 200, 200, "type", (topleft, topleft))
    objects.append(image)
    drawATIImage(image)"""


def checkOnImage(x, y):
    if len(objects) > 0:
        for obj in objects:
            if 50 <= x <= obj.width + 50 and 50 <= y <= obj.height + 50:
                return obj


def drawSelection2(x, y, x2, y2, color):
    top = min(y, y2)
    left = min(x, x2)
    right = max(x, x2)
    bottom = max(y, y2)
    surface = pygame.display.get_surface()
    for x in range(right - left):
        surface.set_at((x + left, top), color)
        surface.set_at((x + left, bottom), color)
    for y in range(bottom - top):
        surface.set_at((left, top + y), color)
        surface.set_at((right, top + y), color)

def makeselection(selection):
    drawSelection2(selection.x, selection.y, selection.prevx, selection.prevy, (255, 255, 255))
    drawSelection2(selection.x, selection.y, selection.newx, selection.newy, (0, 0, 255))

    #rect = (x, y, x2-x, y2-y)
    #pygame.draw.rect(surface, (0,0,255), (x, y, x2-x, y2-y))


def handleMouseinput():
    x, y = pygame.mouse.get_pos()
    imClicked = checkOnImage(x, y)
    if imClicked:
        app.setValueEntry(x - 50, y - 50, imClicked.data[x - 50][y - 50])

def getInput():
    global dragging
    global startx
    global starty
    global newselection
    global isSelectionActive

    for event in pygame.event.get():
        if event.type == QUIT:
            return True
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                startx, starty = pygame.mouse.get_pos()
                if isSelectionActive:
                    drawSelection2(newselection.x, newselection.y, newselection.newx, newselection.newy, newselection.color)
                newselection.set_startpos((startx, starty))
                print("mousedown")
                isSelectionActive = True
                handleMouseinput()
                dragging = True
        elif event.type == MOUSEBUTTONUP:
            print("mouseup")
            if event.button == 1:
                dragging = False

        elif event.type == MOUSEMOTION:
            if dragging:
                x, y = pygame.mouse.get_pos()
                newselection.set_newpos((x, y))
                makeselection(newselection)
        sys.stdout.flush()  # get stuff to the console
    return False


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
        if getInput():
            done = True
        pygame.display.flip()

root = Tk()
pygame.init()
ScreenSize = (700, 400)
surface = pygame.display.set_mode(ScreenSize)
objects = []
app = Window(root)
Done = False

dragging = False
startx = None
starty = None
newselection = Selection()
isSelectionActive = False

editableImage = ATIImage()
originalImage = ATIImage()

if __name__ == '__main__': main()
