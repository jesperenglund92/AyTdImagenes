import math


from tkinter import filedialog, font
from tkinter import *
import pygame
import sys
from pygame.locals import *
from ATIImage import *
import struct
import binascii
import array

root = Tk()
root.geometry('300x200')

editableImage = ATIImage()
originalImage = ATIImage()


class ImageSelection(object):
    def __init__(self, posinitial=[0,0], posfinal=[0, 0]):
        self.posini = posinitial
        self.posfin = posfinal
        self.bottonrigth = [max(posinitial[0], posfinal[0]), max(posinitial[1], posfinal[1])]
        self.bottonleft = [min(posinitial[0], posfinal[0]), min(posinitial[1], posfinal[1])]
        self.active = bool(False)

    def set_final(self, posfin):
        self.posfin = posfin
        self.updatePositions()

    def set_initial(self, posini):
        self.posini = posini
        self.updatePositions()

    def get_final(self):
        return self.posfin

    def get_initial(self):
        return self.posini

    def draw(self):
        pass

    def updatePositions(self):
        self.bottonrigth = [max(self.posini[0], self.posfin[0]), max(self.posini[1], self.posfin[1])]
        self.bottonleft = [min(self.posini[0], self.posfin[0]), min(self.posini[1], self.posfin[1])]

    def is_active(self):
        return self.active

    def set_active(self, active):
        self.active = active
selection = ImageSelection()



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
        fileMenu.add_command(label="New File")
        fileMenu.add_command(label="Load Image", command=openFile)
        fileMenu.add_command(label="Save File", command=saveFile)
        fileMenu.add_command(label="Exit", command=self.exitProgram)
        menu.add_cascade(label="File", menu=fileMenu)
        editMenu = Menu(menu)

        editMenu.add_command(label="Get value", command=getValue)
        editMenu.add_command(label="Edit value")
        editMenu.add_command(label="Select", command=openSelectWindow)
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
        print(self.file.name)
        fuente = font.Font(weight="bold")

        self.lblSelection = Label(self.window, text="Select Raw Size", font=fuente).grid(row=4)
        self.lblInitial = Label(self.window, text="Width").grid(row=5)
        self.lblFinal = Label(self.window, text="Height").grid(row=6)

        self.width = StringVar()
        self.height = StringVar()

        self.txtWidth = Entry(self.window, textvariable=self.x1)
        self.txtHeight = Entry(self.window, textvariable=self.y1)
        self.txtWidth.grid(row=5, column=1)
        self.txtHeight.grid(row=6, column=1)

        self.button = Button(self.window, text="Open raw", command=self.openRawImage)
        self.button.grid(row=8)

    def openRawImage(self):
        print("Width: " + self.txtWidth.get() + " ; Height: "+ self.txtHeight.get())

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
            printNotRawImage()

        if filename.lower().endswith(('.ppm')):
            editableImage.type = '.ppm'
            editableImage.type = '.ppm'
            loadPpm(file)
            printNotRawImage()
        file.close()

    else:
        print("cancelled")

def printNotRawImage ():
    editableImage.topleft = [20, 20]
    originalImage.topleft = [20 + editableImage.width + 20, 20]
    pygame.display.set_mode((40 + editableImage.width * 2, 40 + editableImage.height))
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
            print(image.data[y][x])
            surface.set_at((x + image.topleft[0], y + image.topleft[1]), image.data[y][x])


class SelectionWindow():
    def __init__(self):
        self.selectionWindow = Tk()
        self.selectionWindow.focus_set()

        self.lblSelection = Label(self.selectionWindow, text="Selection").grid(row=4)
        self.lblInitial = Label(self.selectionWindow, text="TopLeft").grid(row=5)
        self.lblFinal = Label(self.selectionWindow, text="TopLeft").grid(row=6)

        self.x1 = StringVar()
        self.y1 = StringVar()
        self.x2 = StringVar()
        self.y2 = StringVar()

        self.txtInitialX = Entry(self.selectionWindow, textvariable=self.x1)
        self.txtInitialY = Entry(self.selectionWindow, textvariable=self.y1)
        self.txtLastX = Entry(self.selectionWindow, textvariable=self.x2)
        self.txtLastY = Entry(self.selectionWindow, textvariable=self.y2)
        self.txtInitialX.grid(row=5, column=1)
        self.txtInitialY.grid(row=5, column=2)
        self.txtLastX.grid(row=6, column=1)
        self.txtLastY.grid(row=6, column=2)

        self.button = Button(self.selectionWindow, text="Select", command=self.printvalues)
        self.button.grid(row=8)

    def printvalues(self):
        print(self.x1.get(), self.y1.get())
        print(self.txtInitialX.get(), self.txtInitialY.get())

def openSelectWindow():
    selectionWindow = SelectionWindow()

    """selectionWindow.geometry("200x200")"""
    #selectionWindow.title("Selection")
    #selectionWindow.focus_set()

    """display = Label(selectionWindow, text="Selection Window")
    display.pack()
    Label(selectionWindow, text="x: ").grid(row=0, column=0)
    Label(selectionWindow, text="y: ").grid(row=1, column=0)"""

    #lblSelection = Label(selectionWindow, text="Selection").grid(row=4)
    #lblInitial = Label(selectionWindow, text="TopLeft").grid(row=5)
    #lblFinal = Label(selectionWindow, text="TopLeft").grid(row=6)


    #txtInitialX = Entry(selectionWindow, textvariable=x1)
    #txtInitialY = Entry(selectionWindow, textvariable=y1)

    """txtLastX = Entry(selectionWindow, textvariable=x2)
    txtLastY = Entry(selectionWindow, textvariable=y2)

    txtInitialX.grid(row=5, column=1)
    txtInitialY.grid(row=5, column=2)
    txtLastX.grid(row=6, column=1)
    txtLastY.grid(row=6, column=2)
    button = Button(selectionWindow, text="Armar selection", command=printvalues)
    button.grid(row=8)
    """

def openRAWWindow():
    rawWindow = Tk()
    rawWindow.title("Select width and heigth")
    rawWindow.focus_set()
    lblSelection = Label(rawWindow, text="width").grid(row=1)
    lblInitial = Label(rawWindow, text="height").grid(row=3)




    """
    txtInitialX.pack()
    txtInitialY.pack()
    
    label = Label(selectionWindow, text="Position: ").grid(row=7, column=0).pack()
    res = Label(selectionWindow).grid(row=7, column = 1)
    res.pack()"""


def getValue():
    pass

class ImageSelection:
    def __init__(self):
        self.posini = [0 , 0]
        self.posfin = [0, 0]

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


def checkOnImage(x, y, blackImage):
    if 50 <= x <= blackImage.width + 50 and 50 <= y <= blackImage.height + 50:
        return True

def handleMouseinput(surface, app):
    x, y = pygame.mouse.get_pos()
    surface = pygame.display.get_surface();
    """if checkOnImage(x, y, blackImage):
        app.setValueEntry(x-50, y-50, blackImage.data[x-50][y-50])"""
    print(surface.get_at((x,y)))

def GetInput(surface, app):
    for event in pygame.event.get():
        if event.type == QUIT:
            return True
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                handleMouseinput(surface, app)
        sys.stdout.flush()  # get stuff to the console
    return False

Done = False


def quit_callback():
    global Done
    Done = True


"""def newBlackImage (width, height, surface):
    data = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append((0,0,0))
        data.append(row)
    return Image(data, width, height, "type", surface, (50, 50))"""

def main():
    # initialise pygame
    pygame.init()
    ScreenSize = (700, 400)
    surface = pygame.display.set_mode(ScreenSize)
    # initialise tkinter

    app = Window(root)

    root.wm_title("Tkinter window")
    root.protocol("WM_DELETE_WINDOW", quit_callback)
    surface.fill((255, 255, 255))
    #blackImage = newBlackImage(300, 300, surface)


    gameframe = 0
    # main loop
    while not Done:
        try:
            app.update()
        except:
            print("dialog error")
        if GetInput(surface, app):  # input event can also comes from diaglog
            break
        # blackImage.draw()
        gameframe += 1
        pygame.display.update()
    app.destroy()


if __name__ == '__main__': main()
