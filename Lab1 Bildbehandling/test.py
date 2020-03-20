import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
from scipy import signal
import imageio
import scipy.misc
import scipy.signal

"""-----------------------------Uppgift 1------------------------------- """

img = np.mean(imageio.imread("teracotta-wall.jpg"), axis=2)
I = (1/9) * np.ones((3, 3))


def convolve_func(img, I):   #Skapar funktion med invärden bildmatrisen och faltningsmatrisen
    output = np.zeros_like(img) #Skapar matris med samma dimenstioner som bildmatrisen med bara nollor
    image_padded = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    image_padded[1:-1, 1:-1] = img
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            output[y, x] = (I * image_padded[y:y + 3, x:x + 3]).sum()
    return output

conv_img = convolve_func(img,I)
done = scipy.signal.convolve2d(img, I, mode="same")

"""fig, ax = plt.subplots(1, 3)
ax[0].imshow(img, cmap = "gray")
ax[1].imshow(conv_img, cmap = "gray")
ax[2].imshow(done, cmap = "gray")
plt.show()"""
"""-----------------------------Uppgift 2------------------------------- """

img_2 = np.mean(imageio.imread("lynn-eyes-halftone.png"), axis=2)


def gaussian_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1] #skapar matrisen utifrån angiven storlek. N/2 i mitten
    g = np.exp(-(x * 2 / float(size) + y * 2 / float(size_y)))
    kernel = g / g.sum() #summerar alla

    return kernel

"""Takes the first row-vector and column-vector to be used to convolve the image-matrix"""
row_vector = gaussian_kernel(5)[1:2]
coloumn_vector = gaussian_kernel(5)[:, [1]]

img_sep = scipy.signal.convolve(scipy.signal.convolve(img_2, row_vector), coloumn_vector) #linjärt separerbar

gauss_img = scipy.signal.convolve2d(img_2, gaussian_kernel(5), mode="same") #använder convolve-funktionen


fig, ax = plt.subplots(1, 3)
ax[0].imshow(img_2, cmap = "gray")
ax[1].imshow(img_sep, cmap = "gray")
ax[2].imshow(gauss_img, cmap = "gray")
plt.show()


"""-----------------------------Uppgift 3------------------------------- """

# Prepare the kernels
a1 = np.matrix([1, 2, 1])
a2 = np.matrix([-1, 0, 1])
Kx = a1.T * a2
Ky = a2.T * a1


# Apply the Sobel operator
Gx = scipy.signal.convolve2d(img, Kx, "same", "symm")
Gy = scipy.signal.convolve2d(img, Ky, "same", "symm")
G = np.sqrt(Gx**2 + Gy**2)

"""
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img, cmap = "gray")
ax[1].imshow(G, cmap = "gray")
plt.show()
"""
"""-----------------------------Uppgift 4------------------------------- """

#plt.imshow(img, cmap = "gray")
#plt.show()

size = 7 #storleken på filterkärnan

def low_pass_kernel(size, size_y=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1] #skapar matrisen utifrån angiven storlek. N/2 i mitten
    g = np.exp(-(x * 2 / float(size) + y * 2 / float(size_y)))
    kernel = g / g.sum() #summerar alla

    return kernel

print(low_pass_kernel(size).sum()) #testar så lågpasskärnan summerar till 1

low_pass = scipy.signal.convolve2d(img, low_pass_kernel(size), mode="same") #faltar matris med orginalbild

filtred_img = img - low_pass #orginalbilden subraherat med gaussade bilden = högpassbild.


fig, ax = plt.subplots(1, 2)
ax[0].imshow(img, cmap = "gray")
ax[1].imshow(filtred_img, cmap = "gray")
plt.show()