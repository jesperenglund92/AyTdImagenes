import sys

sys.path.insert(0, '..')

import pygame
from pygame.locals import *
from pgu import gui

#from newnew import Painter


def open_file_browser(arg):
    d = gui.FileDialog()
    d.connect(gui.CHANGE, handle_file_browser_closed, d)
    d.open()


def handle_file_browser_closed(dlg):
    if dlg.value: input_file.value = dlg.value


app = gui.Desktop()
app.connect(gui.QUIT, app.quit, None)
screen = gui.Container(width=500, height=400)  # , background=(220, 220, 220) )
screen.add(gui.Label("File Dialog Example", cls="h1"), 10, 10)

td_style = {'padding_right': 10}
t = gui.Table()
t.tr()
t.td(gui.Label('File Name:'), style=td_style)
input_file = gui.Input()
t.td(input_file, style=td_style)
b = gui.Button("Browse...")
t.td(b, style=td_style)
b.connect(gui.CLICK, open_file_browser, None)


screen.add(t, 20, 100)


def main():
    app.run(screen)

main()
