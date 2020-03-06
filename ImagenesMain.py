import pygame

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
    screen.set_at((x, y), (255,255,255))
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


#open("BARCO.RAW")