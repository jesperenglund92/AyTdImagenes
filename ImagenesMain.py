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
from tkinter import *
import pygame
import sys
from pygame.locals import *


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        menu = Menu(self.master)
        self.master.config(menu=menu)

        fileMenu = Menu(menu)
        fileMenu.add_command(label="Item")
        fileMenu.add_command(label="Exit", command=self.exitProgram)
        menu.add_cascade(label="File", menu=fileMenu)

        editMenu = Menu(menu)
        editMenu.add_command(label="Undo")
        editMenu.add_command(label="Redo")
        menu.add_cascade(label="Edit", menu=editMenu)

    def exitProgram(self):
        exit()


"""root = Tk()
app = Window(root)"""
"""root.wm_title("Tkinter window")
root.mainloop()"""


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
    #surf.fill((80, 80, 80))
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
    root = Tk()
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
        #clock.tick(100)  # slow it to something slightly realistic
        gameframe += 1
        pygame.display.update()
    app.destroy()

if __name__ == '__main__': main()
# open("BARCO.RAW")
