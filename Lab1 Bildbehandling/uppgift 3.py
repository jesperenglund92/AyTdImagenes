import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
from scipy import signal
import imageio
import time

I = np.mean(imageio.imread("teracotta-wall.jpg"), axis=2)

# Skapar Hx och Hy matriser
Hx = np.matrix([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
Hy = np.matrix([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Faltar med matriserna samt räknar ut beloppet av dessa
Gx = scipy.signal.convolve2d(I, Hx, mode="same")
Gy = scipy.signal.convolve2d(I, Hy, mode="same")
G = np.sqrt(Gx**2 + Gy**2)


fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2)
ax1.imshow(I, cmap = "gray")
ax2.imshow(G, cmap = "gray")
ax3.imshow(Gx, cmap = "gray")
ax4.imshow(Gy, cmap = "gray")
ax1.set_title("Orginal")
ax2.set_title("G")
ax3.set_title("Gx")
ax4.set_title("Gy")
plt.show()

# Gx hittar kanter i x-led och Gy i y-led