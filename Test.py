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
        newSubmenu.add_command(label="circle", command=newWhiteCircle)

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
        self.valueEntry = Entry(master, text="First Name")
        self.valueEntry.grid(row=2, column=1)

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
                self.surface.set_at((x + self.topleft[0], y + self.topleft[0]), self.data[x][y])


def quit_callback():
    global Done
    Done = True


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
    image = Image(data, 200, 200, "type", surface, topleft)
    image.draw()


def checkOnImage(x, y):
    if len(objects) > 0:
        for obj in objects:
            if 50 <= x <= obj.width + 50 and 50 <= y <= obj.height + 50:
                return obj


def handleMouseinput(surface):
    x, y = pygame.mouse.get_pos()
    imClicked = checkOnImage(x, y)
    if imClicked:
        pass
    print(x, y)


def getInput():
    for event in pygame.event.get():
        if event.type == QUIT:
            return True
        if event.type == MOUSEBUTTONDOWN:
            print("hej")
            if event.button == 1:
                handleMouseinput(surface)
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
