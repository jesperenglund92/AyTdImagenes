import math
from threading import Thread
from tkinter import filedialog
from tkinter import *
import pygame
import sys
from pygame.locals import *
from pgu import gui


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        menu = Menu(self.master)
        self.master.config(menu=menu)

        fileMenu = Menu(menu)

        newSubmenu = Menu(fileMenu)
        newSubmenu.add_command(label="Circle", command=newWhiteCircle)
        newSubmenu.add_command(label="Square", command=newWhiteSquare)

        fileMenu.add_cascade(label="New File", menu=newSubmenu)
        fileMenu.add_command(label="Exit", command=self.exitProgram)

        menu.add_cascade(label="File", menu=fileMenu)
        editMenu = Menu(menu)
        editMenu.add_command(label="Edit value")
        menu.add_cascade(label="Edit", menu=editMenu)

        Label(master, text="x: ").grid(row=0, column=0)
        Label(master, text="y: ").grid(row=1, column=0)
        Label(master, text="color: ").grid(row=2, column=0)
        self.xLabel = Label(master, text="0")
        self.xLabel.grid(row=0, column=1)
        self.yLabel = Label(master, text="0")
        self.yLabel.grid(row=1, column=1)
        self.valueEntry = Entry(master)
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
        self.valueEntry.delete(0,END)
        self.valueEntry.insert(0,value)


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
                self.surface.set_at((x + self.topleft[0], y + self.topleft[1]), self.data[x][y])


def quit_callback():
    global Done
    Done = True


def changepixval(x, y, color):
    colorlist = color.split()
    r, g, b = int(colorlist[0]), int(colorlist[1]), int(colorlist[2])
    for obj in objects:
        obj.data[x][y] = (r, g, b)
        obj.draw()


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
    image = Image(data, 200, 200, "type", surface, (topleft, topleft))
    objects.append(image)
    image.draw()


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
    image = Image(data, 200, 200, "type", surface, (topleft, topleft))
    objects.append(image)
    image.draw()


def checkOnImage(x, y):
    if len(objects) > 0:
        for obj in objects:
            if 50 <= x <= obj.width + 50 and 50 <= y <= obj.height + 50:
                return obj


def makeselection(x, y, x2, y2):
    pass


def handleMouseinput():
    x, y = pygame.mouse.get_pos()
    imClicked = checkOnImage(x, y)
    if imClicked:
        app.setValueEntry(x - 50, y - 50, imClicked.data[x - 50][y - 50])

dragging = False
startx = None
starty = None

def getInput():
    global dragging
    global startx
    global starty
    for event in pygame.event.get():
        if event.type == QUIT:
            return True
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                print("mousedown")
                handleMouseinput()
                dragging = True
        elif event.type == MOUSEBUTTONUP:
            print("mouseup")
            if event.button == 1:
                dragging = False
        elif event.type == MOUSEMOTION:
            if dragging:
                x, y = pygame.mouse.get_pos()
                makeselection(startx, starty, x, y)
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
            if getInput():
                done = True
        except:
            print("dialog error")
        pygame.display.flip()


root = Tk()
pygame.init()
ScreenSize = (700, 400)
surface = pygame.display.set_mode(ScreenSize)
objects = []
app = Window(root)

if __name__ == '__main__':
    main()
