from tkinter import filedialog
from tkinter import *
import pygame
import sys
from pygame.locals import *
import struct
import binascii
import array

root = Tk()


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
        for x in range(self.height):
            for y in range(self.width):
                self.surface.set_at((x + self.topleft[0], y + self.topleft[0]), self.data[x][y])



def checkOnImage(x, y, blackImage):
    if 50 <= x <= blackImage.width + 50 and 50 <= y <= blackImage.height + 50:
        return True


def handleMouseinput(surface, app, blackImage):
    x, y = pygame.mouse.get_pos()
    if checkOnImage(x, y, blackImage):
        app.setValueEntry(x-50, y-50, blackImage.data[x-50][y-50])
    print(x, y)


def GetInput(surface, app, blackImage):
    for event in pygame.event.get():
        if event.type == QUIT:
            return True
        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                handleMouseinput(surface, app, blackImage)
        sys.stdout.flush()  # get stuff to the console
    return False


Done = False


def quit_callback():
    global Done
    Done = True


def newBlackImage (width, height, surface):
    data = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append((0,0,0))
        data.append(row)
    return Image(data, width, height, "type", surface, (50, 50))

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
    blackImage = newBlackImage(300, 300, surface)


    gameframe = 0
    # main loop
    while not Done:
        if GetInput(surface, app, blackImage):  # input event can also comes from diaglog
            break
        try:
            app.update()
        except:
            print("dialog error")
        blackImage.draw()
        gameframe += 1
        pygame.display.update()
    app.destroy()


if __name__ == '__main__': main()
