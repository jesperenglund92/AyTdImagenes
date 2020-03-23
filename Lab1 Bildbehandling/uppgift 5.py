import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import imageio
from scipy import signal
import time


I = np.mean(imageio.imread("teracotta-wall.jpg"), axis=2)


#absolute = np.abs(freq)
#log = np.log(absolute)

def high_pass(I, N):
    I_freq = np.fft.fftshift(np.fft.fft2(I))        #Gör om bilden till frekvensdomänen
    x, y = np.shape(I_freq)
    filter_matrix = np.ones((x, y))                 #Skapar en matris med ettor, lika stor som bilden
    for row in range(int((x/2)-N), int((x/2)+N+1)):
        for col in range(int((y/2)-N), int((y/2)+N+1)):
            filter_matrix[row][col] = 0             #Skapar en N*N ruta med nollor i mitten av matrisen

    high_pass_freq = I_freq * filter_matrix         #Faltar i frekvensdomänen
    high_pass_spatial = np.fft.ifft2(np.fft.fftshift(high_pass_freq))   #Transformerar till spatialdomänen
    high_pass_spatial_abs = np.absolute(high_pass_spatial)

    return high_pass_spatial_abs


def low_pass(I, N):
    I_freq = np.fft.fftshift(np.fft.fft2(I))        #Gör om bilden till frekvensdomänen
    x, y = np.shape(I_freq)
    filter_matrix = np.zeros((x, y))                #Skapar en matris med nollor, lika stor som bilden
    for row in range(int((x/2)-N), int((x/2)+N+1)):
        for col in range(int((y/2)-N), int((y/2)+N+1)):
            filter_matrix[row][col] = 1             #Skapar en N*N ruta med ettor i mitten av matrisen

    low_pass_freq = I_freq * filter_matrix          #Faltar i frkvensdomänen
    low_pass_spatial = np.fft.ifft2(np.fft.fftshift(low_pass_freq))     #Transformerar till spatialdomänen
    low_pass_spatial_abs = np.absolute(low_pass_spatial)

    return low_pass_spatial_abs


fig, ax = plt.subplots(1, 2)
ax[0].imshow(I, cmap="gray")

ax[1].imshow(high_pass(I, 20), cmap="gray")
#ax[1].imshow(low_pass(I, 20), cmap="gray")
plt.show()


