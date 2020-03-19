from tkinter import filedialog, font
import pygame
from pygame.locals import *
from ATIImage import *
from classes import *

class PPMException(Exception):
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

        self.screenX = 0
        self.screenY = 0

        file_menu = Menu(menu)

        fileSubmenu = Menu(file_menu)
        fileSubmenu.add_command(label="circle", command=newWhiteCircle)
        fileSubmenu.add_command(label="Square", command=newWhiteSquare)

        file_menu.add_cascade(label="New File", menu=fileSubmenu)
        file_menu.add_command(label="Load Image", command=openFile)
        file_menu.add_command(label="Save File", command=saveFile)
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
        menu.add_cascade(label="View", menu = view_menu)
        edit_menu.add_command(label="Copy selection", command=copy_selection)


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
                                                             self.screenX, self.screenY,
                                                             self.valueEntry.get()))
        self.changebtn.grid(row=2, column=2)
        Label(master, text="Pixel amount: ").grid(row=3, column=0)
        self.pixel_amount = Label(master, text="0")
        self.pixel_amount.grid(row=3, column=1)
        Label(master, text="Grayscale average: ").grid(row=4, column=0)
        self.gray_avg = Label(master, text="0")
        self.gray_avg.grid(row=4, column=1)

    def exitProgram(self):
        exit()

    def display_pixelval(self, x, y, value, screenx, screeny):
        self.xLabel['text'] = x
        self.yLabel['text'] = y
        self.valueEntry.delete(0, END)
        self.valueEntry.insert(0, value)
        self.screenX = screenx
        self.screenY = screeny

    def display_gray_pixamount(self, amount, grayavg):
        self.pixel_amount['text'] = amount
        self.gray_avg['text'] = grayavg

    def copy_window(self):
        pass

    def operations_window(self):
        pass

    def threshold_window(self):
        pass

    def equalization_window(self):
        pass

    def make_negative(self):
        pass

    def histogram_window(self):
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
        file.close()

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

    class _Threshold_window():
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


def set_image(image, data, width, height, type, topleft, editable):
    image.data = data
    image.width = width
    image.height = height
    image.type = type
    image.topleft = topleft
    image.editable = editable
    image.values_set = True


def imgdata_inselection(img):
    #function returning the colordata of in-image within current selection.
    data = []
    yiterator = 0
    for i in range(abs(newselection.newy - newselection.y)):
        row = []
        if newselection.newy < newselection.y:
            y = newselection.newy
        else:
            y = newselection.y
        xiterator = 0
        for j in range(abs(newselection.newx - newselection.x)):
            if newselection.newx < newselection.x:
                x = newselection.newx
            else:
                x = newselection.x
            if img.collidepoint(x + j, y + i):
                row.append(img.get_at_screenpos(j + x, y + i))
                xiterator += 1

        if xiterator > 0:
            yiterator += 1
        if len(row) > 0:
            data.append(row)
    return data


def copy_selection():
    #function copying and drawing a copy of selected part of original image.
    for img in images:
        if img.values_set:
            if not img.editable:
                data = imgdata_inselection(img)
                image = ATIImage(data, len(data[0]), len(data), "type", (300, 50), True, True)
                images.append(image)
                drawATIImage(image)


def loadPpm(file):
    count = 0
    height = 0
    width = 0
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
    height = 0
    width = 0
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
            drawImages()

        if filename.lower().endswith(('.ppm')):
            editableImage.type = '.ppm'
            editableImage.type = '.ppm'
            loadPpm(file)
            drawImages()
        file.close()

    else:
        print("cancelled")


def drawImages():
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


def changepixval(x, y, screenx, screeny, color):
    colorlist = color.split()
    r, g, b = int(colorlist[0]), int(colorlist[1]), int(colorlist[2])
    for img in images:
        if img.values_set:
            if img.collidepoint(screenx, screeny):
                if img.editable:
                    img.set_at((x, y), (r, g, b))
                    surface = pygame.display.get_surface()
                    surface.set_at((screenx, screeny), (r, g, b))


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
    set_image(originalImage, data, 200, 200, "type", (topleft, topleft), False)
    drawATIImage(originalImage)


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
    set_image(originalImage, data, 200, 200, "type", (topleft, topleft), False)
    drawATIImage(originalImage)


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


def rgbtograyscale(pixelcol):
    return pixelcol[0] * 0.3 + pixelcol[1] * 0.59 + pixelcol[2] * 0.11


def get_gray_pixamount(img):
    if img.values_set:
        data = imgdata_inselection(img)
        if len(data) > 0:
            width = len(data[0])
            height = len(data)
            pixelamount = width * height
            sum = 0
            for row in data:
                for col in row:
                    sum += rgbtograyscale(col)
            avggray = round(sum/pixelamount, 2)
            app.display_gray_pixamount(pixelamount, avggray)


def makeselection(selection):
    drawSelection(selection.x, selection.y, selection.prevx, selection.prevy, (255, 255, 255))
    for img in images:
        drawATIImage(img)
        if img.collidepoint(selection.newx, selection.newy):
            get_gray_pixamount(img)
    drawSelection(selection.x, selection.y, selection.newx, selection.newy, (0, 0, 255))

    # rect = (x, y, x2-x, y2-y)
    # pygame.draw.rect(surface, (0,0,255), (x, y, x2-x, y2-y))


def handleMouseinput():
    x, y = pygame.mouse.get_pos()
    for img in images:
        if img.collidepoint(x, y):
            app.display_pixelval(x - img.topleft[0], y - img.topleft[1], img.get_at_screenpos(x, y), x, y)



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
                    drawSelection(newselection.x, newselection.y, newselection.newx, newselection.newy, (255, 255, 255))
                newselection.set_startpos((startx, starty))
                isSelectionActive = True
                handleMouseinput()
                dragging = True
        elif event.type == MOUSEBUTTONUP:
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
ScreenSize = (800, 400)
surface = pygame.display.set_mode(ScreenSize)
images = [] #list of images, in case we need to be more flexible than just one editable and one original image, possible to add more.
app = Window(root)
Done = False

dragging = False
startx = None
starty = None
newselection = Selection()
isSelectionActive = False

editableImage = ATIImage(editable=True)
originalImage = ATIImage()

images.append(editableImage)
images.append(originalImage)

if __name__ == '__main__': main()
