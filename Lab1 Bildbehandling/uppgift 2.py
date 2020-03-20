import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
from scipy import signal
import time
import imageio

# Uppgift 2

N = 10
I = np.mean(imageio.imread("lynn-eyes-halftone.png"), axis=2)

sigma = N/6

def gaussblur_a(I, N):
    # Skapar en gauss-filterkärna och dividerar med sin summa, så att kärnan summerar till 1
    x, y = np.mgrid[-N:N+1, -N:N+1]
    gausskernel = np.exp(-(x**2 + y**2)/(2 * sigma**2))
    core = gausskernel / np.sum(gausskernel)

    J = scipy.signal.convolve2d(I, core, mode="same")       # Kör faltning och returnerar en lågpassfiltrerad bild

    return J


def gaussblur_b(I, N):
    # Skapar en 1 x 2N+1 -matris i gauss-form och transponerar den
    x = np.array([range(-N, N+1)])
    gauss = np.exp(-(x ** 2) / (2 * sigma ** 2))
    gausskernel_x = gauss / np.sum(gauss)
    gausskernel_y = np.transpose(gausskernel_x)

    # Faltar i x- och sedan i y-led
    J = scipy.signal.convolve(scipy.signal.convolve(I, gausskernel_x, mode="same"), gausskernel_y, mode="same")

    return J


t = time.time() # Ger tiden just nu
gaussblur_a(I, N)
tid_det_tog = time.time() - t # Ger tiden det tog att köra faltning_i_2d()
print(tid_det_tog)

t = time.time() # Ger tiden just nu
gaussblur_b(I, N)
tid_det_tog = time.time() - t # Ger tiden det tog att köra faltning_i_2d()
print(tid_det_tog)


"""plt.imshow(gaussblur_a(I, N), cmap="gray")
plt.subplots()"""
plt.imshow(gaussblur_b(I, N), cmap="gray")
plt.subplots()
plt.imshow(I, cmap="gray")
plt.show()










#convolve_x = scipy.signal.convolve(I, gausskernel_x, mode="same")
    #convolve_y = scipy.signal.convolve(I, gausskernel_y, mode="same")
    #J = convolve_x * convolve_y
    #convolve_y = scipy.signal.convolve2d(convolve_x, gausskernel_y)