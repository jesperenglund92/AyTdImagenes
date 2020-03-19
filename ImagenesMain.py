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

        file_menu = Menu(menu)

        new_submenu = Menu(file_menu)
        new_submenu.add_command(label="circle", command=newWhiteCircle)
        new_submenu.add_command(label="Square", command=newWhiteSquare)

        file_menu.add_cascade(label="New File", menu=new_submenu)
        file_menu.add_command(label="Load Image", command=self.openFile)
        file_menu.add_command(label="Save File", command=self.saveFile, state="disabled")
        file_menu.add_command(label="Exit", command=self.exitProgram)

        menu.add_cascade(label="File", menu=file_menu)
        edit_menu = Menu(menu)
        edit_menu.add_command(label="Copy", command=self.copy_window)
        edit_menu.add_command(label="Operations", command=self.operations_window)
        edit_menu.add_command(label="Threshold Image", command=self.threshold_window)
        edit_menu.add_command(label="Equalize Image", command=self.equalization_window)
        edit_menu.add_command(label="Negative", command=self.make_negative)
        menu.add_cascade(label="Edit", menu=edit_menu)

        view_menu = Menu(menu)
        view_menu.add_command(label="HSV Color")
        view_menu.add_command(label="Histogram", command=self.histogram_window)
        menu.add_cascade(label="View", menu=view_menu)

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



        Label(master, text="Region seleccionada: ").grid(row=4, column=0)
        Label(master, text="Grey Average: ").grid(row=5, column=0)
        Label(master, text="Red Average: ").grid(row=6, column=0)
        Label(master, text="Green Average ").grid(row=7, column= 0)
        Label(master, text="Blue Average ").grid(row=8, column= 0)

        self.selection_pixel_count = Label(master, text="0")
        self.selection_pixel_count.grid(row=4, column=2)
        self.grey_pixel_average = Label(master, text="0")
        self.grey_pixel_average.grid(row=5, column=2)

        self.red_pixel_average = Label(master, text="0")
        self.red_pixel_average.grid(row=6, column=2)

        self.green_pixel_average = Label(master, text="0")
        self.green_pixel_average.grid(row=7, column=2)

        self.blue_pixel_average = Label(master, text="0")
        self.blue_pixel_average.grid(row=8, column=2)


    def exitProgram(self):
        pygame.display.quit()
        pygame.quit()
        exit()

    def setValueEntry(self, x, y, value):
        self.xLabel['text'] = x
        self.yLabel['text'] = y
        self.valueEntry.delete(0, END)
        self.valueEntry.insert(0, value)

    def copy_window(self):
        window = self.__CopyWindow()

    class __CopyWindow():
        def __init__(self):
            self.window = Tk()
            self.window.focus_set()
            self.window.title("Copy Image")
            pass

    def operations_window(self):
        window = self.__OperationsWindow()

    class __OperationsWindow():
        def __init__(self):
            self.window = Tk()
            self.window.focus_set()
            self.window.title("Copy Image")
            pass

    def threshold_window(self):
        window = self._Threshold_window()

    class _ThresholdWindow():
        def __init__(self):
            self.window = Tk()
            self.window.focus_set()
            self.window.title("Threshold Image")
            pass

    def equalization_window(self):
        window = self.__EqualizationWindow()

    class __EqualizationWindow():
        def __init__(self):
            self.window = Tk()
            self.window.focus_set()
            self.window.title("Threshold Image")

    def make_negative(self):
        editableImage.negative()

    def histogram_window(self):
        window = self.__HistogramWindow()

    class __HistogramWindow():
        def __init__(self):
            self.window = Tk()
            self.window.focus_set()
            self.window.title("Histogram window")

    def loadPpm(self, file):
        global editableImage
        global originalImage
        count = 0
        while count < 3:
            line = file.readline()
            if line[0] == '#':  # Ignore comments
                continue
            count = count + 1
            if count == 1:  # Magic num info
                magicNum = line.strip()
                if magicNum != 'P3' or magicNum != 'P6':
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

    def savePpm(self, file):
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

    def loadPgm(self, file):
        global editableImage
        global originalImage
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

    def savePgm(self, file):
        image = editableImage.get_data()
        width = editableImage.get_size()[0]
        height = editableImage.get_size()[1]
        ## TODO: Write headers

        for y in range(height):
            for x in range(width):
                file.write(int.to_bytes(image[x][y], byteorder="big"))
        pass

    class __RawWindow:
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
            global editableImage
            global originalImage

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
            drawImages()
            file.close()

    def loadRaw(self, file):
        window = self.__RawWindow(file)

    def saveRaw(self, file):
        image = editableImage.get_data()
        width = editableImage.get_size()[0]
        height = editableImage.get_size()[1]

        # surface = pygame.display.set_mode((width, height))
        for y in range(height):
            for x in range(width):
                color = int.to_bytes(image[x][y], byteorder="big")
                file.write(color)
        pass

    def openFile(self):
        global editableImage
        global originalImage

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
                self.loadRaw(file)
            if filename.lower().endswith(('.pgm')):
                editableImage.type = '.pgm'
                editableImage.type = '.pgm'
                self.loadPgm(file)
                drawImages()

            if filename.lower().endswith(('.ppm')):
                editableImage.type = '.ppm'
                editableImage.type = '.ppm'
                self.loadPpm(file)
                drawImages()
            file.close()

        else:
            print("cancelled")

    def saveFile(self):
        file = filedialog.asksaveasfile(mode='w', defaultextension=editableImage.get_type())
        if file:
            if file.name.lower().endswith(('.raw')):
                self.saveRaw(file)
            if file.name.lower().endswith(('.pgm')):
                self.savePgm(file)
            if file.name.lower().endswith(('.ppm')):
                self.savePpm(file)
        pass

    def openRAWWindow(self):
        rawWindow = Tk()
        rawWindow.title("Select width and heigth")
        rawWindow.focus_set()
        lblSelection = Label(rawWindow, text="width").grid(row=1)
        lblInitial = Label(rawWindow, text="height").grid(row=3)


def drawImages():
    global editableImage
    global originalImage
    editableImage.topleft = [20, 20]
    # originalImage.topleft = [40 + originalImage.width, 20]
    originalImage.set_top_left([40 + originalImage.width, 20])
    editableImage.active = True
    originalImage.active = False

    pygame.display.set_mode((60 + editableImage.width * 2, 40 + editableImage.height))
    drawATIImage(editableImage)
    drawATIImage(originalImage)



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


def quit_callback():
    global Done
    Done = True


def changepixval(x, y, color):
    colorlist = color.split()
    r, g, b = int(colorlist[0]), int(colorlist[1]), int(colorlist[2])

    editableImage.data[y - 1][x - 1] = (r, g, b)
    editedX = x + editableImage.get_top_left()[0]
    editedY = y + editableImage.get_top_left()[1]
    surface = pygame.display.get_surface()
    surface.set_at((editedX, editedY), (r, g, b))

    #for obj in objects:
    #    obj.data[x][y] = (r, g, b)
    #    drawATIImage(obj)


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
    image = ATIImage(data, 200, 200, "type", (topleft, topleft))
    objects.append(image)
    drawATIImage(image)


def drawSelection(x, y, x2, y2, color):
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

def drawPreImageSelection(selection):
    tl = selection.get_prev_top_left()
    br = selection.get_prev_botton_right()

    top = tl[1]
    left = tl[0]
    bottom = br[1]
    right = br[0]
    image = get_image_by_id(selection.image)
    surface = pygame.display.get_surface()
    for x in range(right - left):
        surface.set_at((x + left, top), image.get_at_display((x + left, top)))
        surface.set_at((x + left, bottom), image.get_at_display((x + left, bottom)))
    for y in range(bottom - top):
        surface.set_at((left, top + y), image.get_at_display((left, top + y)))
        surface.set_at((right, top + y), image.get_at_display((right, top + y)))


def drawImageSelection(selection):
    tl = selection.get_top_left()
    br = selection.get_botton_right()

    top = tl[1]
    left = tl[0]
    bottom = br[1]
    right = br[0]
    image = get_image_by_id(selection.image)

    surface = pygame.display.get_surface()
    for x in range(right - left):
        surface.set_at((x + left, top), image.get_at_display((x + left, top)))
        surface.set_at((x + left, bottom), image.get_at_display((x + left, bottom)))
    for y in range(bottom - top):
        surface.set_at((left, top + y), image.get_at_display((left, top + y)))
        surface.set_at((right, top + y), image.get_at_display((right, top + y)))


def get_image_by_id(id):
    if id == 0:
        return editableImage
    if id == 1:
        return originalImage
    raise Exception("Not valid image")

def is_click_in_images(pos):
    # Tengo posiciones de Top Left y botton rigth. Puedo consultar con cualquier imagen
    if editableImage.in_display_image(pos):
        return 0
    if originalImage.in_display_image(pos):
        return 1
    return -1


def makeselection(selection):
    drawPreImageSelection(selection)
    drawSelection(selection.x, selection.y, selection.newx, selection.newy, (0, 0, 255))

    # rect = (x, y, x2-x, y2-y)
    # pygame.draw.rect(surface, (0,0,255), (x, y, x2-x, y2-y))


def checkOnImage(x, y):
    if len(objects) > 0:
        for obj in objects:
            if 50 <= x <= obj.width + 50 and 50 <= y <= obj.height + 50:
                return obj


def handleMouseinput(mouse_pos, image_click):
    image = get_image_by_id(image_click)
    pos_display = image.get_pos_display(mouse_pos)
    app.setValueEntry(pos_display[0], pos_display[1], image.get_at_display(mouse_pos))

    #imClicked = checkOnImage(x, y)
    #if imClicked:
    #    app.setValueEntry(x - 50, y - 50, imClicked.data[x - 50][y - 50])

def updateSelectionValues(selection):
    app.selection_pixel_count["text"] = selection.get_pixel_count()
    image_id = selection.image
    if image_id != -1:
        image_selected = get_image_by_id(image_id)
        if image_selected.image_color_type() == 'g':
            app.grey_pixel_average["text"] = image_selected\
                .get_grey_average_display(selection.get_top_left(), selection.get_botton_right())
        else:
            app.red_pixel_average["text"] = image_selected\
                .get_red_average_display(selection.get_top_left(), selection.get_botton_right())
            app.green_pixel_average["text"] = image_selected\
                .get_green_average_display(selection.get_top_left(), selection.get_botton_right())
            app.blue_pixel_average["text"] = image_selected\
                .get_blue_average_display(selection.get_top_left(), selection.get_botton_right())
    return


def getInput():
    global dragging
    global startx
    global starty
    global newselection
    global isSelectionActive
    global lastaction

    for event in pygame.event.get():
        if event.type == QUIT:
            return True
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:

                print("mousedown")

                mouse_positon = pygame.mouse.get_pos()
                image_click = is_click_in_images(mouse_positon)

                if editableImage.active and image_click != -1: # and is_click_in_images(startx, starty):
                    if isSelectionActive:
                        drawImageSelection(newselection)

                    newselection.set_startpos(mouse_positon)
                    newselection.set_image(image_click)

                    updateSelectionValues(newselection)

                    dragging = True
                    isSelectionActive = True

                    handleMouseinput(mouse_positon, image_click)


                lastaction = "mousedown"
        elif event.type == MOUSEBUTTONUP:
            print("mouseup")
            if event.button == 1:
                if lastaction != "mousemotion":
                    isSelectionActive = False
                    updateSelectionValues(newselection)
                dragging = False
            lastaction = "mouseup"
        elif event.type == MOUSEMOTION:
            if dragging:
                if is_click_in_images(pygame.mouse.get_pos()) == newselection.image:
                    newselection.set_newpos(pygame.mouse.get_pos())
                    makeselection(newselection)
                    updateSelectionValues(newselection)
                else:
                    pass
            lastaction="mousemotion"
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
ScreenSize = (1, 1)
surface = pygame.display.set_mode(ScreenSize)
objects = []
app = Window(root)
Done = False

dragging = False
startx = None
starty = None
newselection = Selection()
isSelectionActive = False
lastaction = None

editableImage = ATIImage()
originalImage = ATIImage()

if __name__ == '__main__': main()
