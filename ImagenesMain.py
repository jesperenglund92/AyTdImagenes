"""import pygame
pygame.display.init()
pygame.display.set_caption("Test")
screen = pygame.display.set_mode((1100, 600))

class Image:
    def __init__(self, data, width, height, type):
        self.data = data
        self.width = width
        self.height = height
        self.type = type
def createObjects():
    pass

def drawImages():
    screen.set_at((100, 100), (255, 255, 255))

def handleMouseinput():
    x, y = pygame.mouse.get_pos()
    screen.set_at((x, y), (255, 255, 255))
    pass

def main():
    running = True
    screen.fill([0, 0, 0])
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    handleMouseinput()
        createObjects()
        drawImages()
        pygame.display.update()
main()
"""
from tkinter import filedialog
from tkinter import *
import pygame
import sys
from pygame.locals import *

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


def partition(s, ch):
    if (ch in s):
        i = s.index(ch)
        return (s[0:i], s[i], s[i + 1:])
    else:
        return (s, None, None)

def strip_comments(s):
    #
    #  Works in 2.5.1, but not in older versions
    #
    #  (rval, junk1, junk2) = s.partition("#")
    (rval, junk1, junk2) = partition(s, "#")
    return rval.rstrip(" \t\n")


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
    img = []
    buf = file.read()
    elem = buf.split()
    """if len(elem) != width * height:
        print('Error in number of pixels')
        exit()"""
    surface = pygame.display.set_mode((width, height))
    for y in range(height):
        tmpList = []
        for x in range(width):
            tmpList.append([elem[(y * width + x) * 3],
                            elem[(y * width + x) * 3 + 1],
                            elem[(y * width + x) * 3 + 2]])
            print([elem[(y * width + x) * 3],
                    elem[(y * width + x) * 3 + 1],
                    elem[(y * width + x) * 3 + 2]])
            surface.set_at((x, y), (image[x][0], image[x][0]))
        img.append(tmpList)

    """
    magic = strip_comments(file.readline())
    # Magic Number
    if magic != "P6":
        raise PPM_Exception('The file being loaded does not appear to be a valid ASCII PPM file')

    # (width, sep, height)
    dimensions = strip_comments(file.readline())
    (width, sep, height) = partition(dimensions, " ")

    width = int(width)
    height = int(height)

    if (width <= 0) or (height <= 0):
        raise PPM_Exception("The file being loaded does not appear to have valid dimensions (" + str(
            width) + " x " + str(height) + ")")

    # Depth of color
    depth = file.readline()
    depth = int(strip_comments(depth))
    if max != 255:
        sys.stderr.write("Warning: PPM file does not have a maximum value of 255.  Image may not be handled correctly.")

    color_list = []
    for line in file:
        line = strip_comments(line)
        color_list += line.split(" ")
    image = []
    surface = pygame.display.set_mode((width, height))"""
    """for x in range(0, width):
        image.append([])
        for y in range(0, height):
            image[x].append([color_list[(y * width + x) * 3],
                             color_list[(y * width + x) * 3 + 1],
                             color_list[(y * width + x) * 3 + 2]])
            print(image[x])
    """        """#surface.set_at((x, y), (image[x][0], image[x][0]))"""
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
