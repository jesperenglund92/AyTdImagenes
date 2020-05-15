import numpy as np

r = np.array(((1, 2, 3),
              (4, 5, 6),
              (7, 8, 9)
              ))

g = np.array(((2, 2, 2),
              (2, 2, 2),
              (2, 2, 2)
              ))

b = np.array(((3, 3, 3),
              (3, 3, 3),
              (3, 3, 3)
              ))

r2 = np.array((1, 1, 1,
               1, 1, 1,
               1, 1, 1
               ))

g2 = np.array((2, 2, 2,
               2, 2, 2,
               2, 2, 2
               ))

b2 = np.array((3, 3, 3,
               3, 3, 3,
               3, 3, 3
               ))

rgb = np.dstack((r, g))

rgb = np.dstack((rgb, b))


def susan():
    mask = np.zeros((7, 7))
    ones = np.ones((7, 7))
    mask[:, 2:-2] = [1, 1, 1]
    mask[1:-1, 1:-1] = [1, 1, 1, 1, 1]
    mask[2:-2, :] = [1, 1, 1, 1, 1, 1, 1]
    print(mask)
    mask += ones
    print(mask)

from tkinter import *



class Test:
    def __init__(self):
        master = Tk()
        Label(master, text="Your sex:").grid(row=0, sticky=W)
        self.var1 = IntVar()
        Checkbutton(master, text="male", variable=self.var1).grid(row=1, sticky=W)
        self.var2 = IntVar()
        Checkbutton(master, text="female", variable=self.var2).grid(row=2, sticky=W)
        Button(master, text='Quit', command=master.quit).grid(row=3, sticky=W, pady=4)
        Button(master, text='Show', command=self.var_states).grid(row=4, sticky=W, pady=4)
        mainloop()

    def var_states(self):
        print("male: %d,\nfemale: %d" % (self.var1.get(), self.var2.get()))

Test()