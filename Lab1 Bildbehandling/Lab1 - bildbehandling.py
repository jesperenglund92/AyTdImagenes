import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
from scipy import signal
import imageio

#Uppgift 1

#Core-matrisen
core = np.ones((3, 3)) * (1/9)

#Gör om bilden från 3d till 2d
I = np.mean(imageio.imread("teracotta-wall.jpg"), axis = 2)

#Lägger till en ram av nollor
I_pad = np.pad(I, 1, "constant", )

#storleken på I
height, width = I_pad.shape

new_I = np.ones((height, width))

#utför element-vis multiplikation och ersätter värden i new_I
for row in range(1, height-1):
    for i in range(1, width-1):
        temp = core * I_pad[row-1:row+2, i-1:i+2]
        new_I[row][i] = np.sum(temp)


#plottar orginalet
plt.imshow(I, cmap="gray")
plt.subplots()

#plottar manuell faltning
plt.imshow(new_I, cmap="gray")
plt.subplots()

#plottar automatisk faltning
auto_conv = scipy.signal.convolve(I, core, mode="same")
plt.imshow(auto_conv, cmap="gray")

plt.show()



