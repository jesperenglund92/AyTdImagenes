import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
from scipy import signal
import imageio
import time


N = 8
sigma = N/6
I = np.mean(imageio.imread("teracotta-wall.jpg"), axis=2)


def low_pass_filter(I, N):
    # Skapar en lågpassfilter-kärna och returnerar en filtrerad bild
    x, y = np.mgrid[-N:N+1, -N:N+1]
    gausskernel = np.exp(-(x**2 + y**2)/(2 * sigma**2))
    core = gausskernel / np.sum(gausskernel)
    J = scipy.signal.convolve2d(I, core, mode="same")
    return J


# Subtraherar den lågpass-filtrerade bilden från orginalet
high_pass_img = I - low_pass_filter(I, N)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(I, cmap = "gray")
ax[1].imshow(high_pass_img, cmap = "gray")
ax[0].set_title("Orginal")
ax[1].set_title("High pass")

plt.show()