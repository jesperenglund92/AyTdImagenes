import math
from threading import Thread
from tkinter import filedialog
from tkinter import *
import pygame
import sys
from pygame.locals import *
from pgu import gui


app = gui.Desktop()
app.connect(gui.QUIT, app.quit, None)
screen = gui.Container(width=500, height=400)
pygame.init()
ScreenSize = (700, 400)
surface = pygame.display.set_mode(ScreenSize)
objects = []


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


#app = Window(root)
input_file = gui.Input()

def open_file_browser(arg):
    d = gui.FileDialog()
    d.connect(gui.CHANGE, handle_file_browser_closed, d)
    d.open()


def handle_file_browser_closed(dlg):
    if dlg.value: input_file.value = dlg.value

def main():
    # initialise pygame
    """pygame.init()
    ScreenSize = (700, 400)
    surface = pygame.display.set_mode(ScreenSize)"""
    # initialise tkinter

    #root.wm_title("Tkinter window")
    surface.fill((255, 255, 255))
    b = gui.Button("Browse...")
    b.connect(gui.CLICK, open_file_browser, None)
    #surface.blit(surface, b)
    #thread = Thread(target=updatetk, args=())
    #thread.start()
    #newWhiteCircle()
    # main loop
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == QUIT:
                print("hejsan")
                done = True
            if event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    print("hej")
                    open_file_browser(0)
                    #handleMouseinput(surface)
        """try:
            getInput()
        except:
            print("dialog error")"""
        pygame.display.update()


if __name__ == '__main__':
    main()
