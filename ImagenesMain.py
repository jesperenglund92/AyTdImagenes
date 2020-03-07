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
        fileMenu.add_command(label="Item")
        fileMenu.add_command(label="Exit", command=self.exitProgram)
        menu.add_cascade(label="File", menu=fileMenu)

        editMenu = Menu(menu)
        editMenu.add_command(label="Undo")
        editMenu.add_command(label="Redo")
        menu.add_cascade(label="Edit", menu=editMenu)

    def exitProgram(self):
        exit()


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
    def __init__(self, data, width, height, type):
        self.data = data
        self.width = width
        self.height = height
        self.type = type


def handleMouseinput(surface):
    x, y = pygame.mouse.get_pos()
    surface.set_at((x, y), (255, 255, 255))
    pass


def Draw(surf):
    # Clear view
    # surf.fill((80, 80, 80))
    pygame.display.flip()


def GetInput(surface):
    for event in pygame.event.get():
        if event.type == QUIT:
            return True
        if event.type == KEYDOWN:
            print(event)
        if event.type == MOUSEBUTTONDOWN:
            print(event)
            if event.button == 1:
                handleMouseinput(surface)
        sys.stdout.flush()  # get stuff to the console
    return False


Done = False


def quit_callback():
    global Done
    Done = True


def main():
    # initialise pygame
    pygame.init()
    ScreenSize = (200, 200)
    surface = pygame.display.set_mode(ScreenSize)
    # initialise tkinter

    app = Window(root)
    root.wm_title("Tkinter window")
    root.protocol("WM_DELETE_WINDOW", quit_callback)
    """main_dialog = tkinter.Frame(root)
    main_dialog.pack()"""
    # start pygame clock
    clock = pygame.time.Clock()
    gameframe = 0
    # main loop
    while not Done:
        try:
            app.update()
        except:
            print("dialog error")
        if GetInput(surface):  # input event can also comes from diaglog
            break
        gameframe += 1
        pygame.display.update()
    app.destroy()


if __name__ == '__main__': main()
