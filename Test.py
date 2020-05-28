import numpy as np
import matplotlib.pyplot as plt
from cv2 import *
from skimage import data, io, filters, feature
from skimage.transform import probabilistic_hough_line

from PIL import Image

r = np.array(((1, 2, 3),
              (4, 5, 6),
              (7, 8, 9)
              ))

print(r.shape)

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


def hough_line(img, vote_thresh, theta_step=1, rho_step=1):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, theta_step))
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width ** 2 + height ** 2)))  # max_dist
    rhos = np.linspace(-diag_len, diag_len, int((diag_len * 2/rho_step) + 1))
    # Save some values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_indices, x_indices = np.nonzero(img)  # (row, col) indices for borders
    lines = []
    # Vote in the hough accumulator
    for i in range(len(x_indices)):
        x = x_indices[i]
        y = y_indices[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + diag_len
            accumulator[rho, t_idx] += 1

    # add to lines if it has more than vote_thresh votes
    for rho in range(len(accumulator)):
        for theta in range(len(accumulator[0])):
            if accumulator[rho][theta] >= vote_thresh:
                lines.append([rho * rho_step - diag_len, theta * theta_step])
    # convert thetas back to radians
    for i in range(len(lines)):
        lines[i][1] -= 90
        lines[i][1] = lines[i][1] * np.pi/180

    # iterate through pixels in image. If the (x, y) point satisfies Hough transform function, mark as line (set to 1).
    fin_lines = np.zeros(img.shape)
    for y in range(len(img)):
        for x in range(len(img[0])):
            for rho, theta in lines:
                eq = int(round(x * np.cos(theta) + y * np.sin(theta)))
                if eq == rho:
                    fin_lines[y][x] = 1
    return fin_lines


def create_bin_square_edges(size):
    image = np.zeros((size, size))
    index1 = int(size/4 - 1)
    index2 = size - int(size/4)
    image[index1:index2, index1] = 1
    image[index1:index2, index2] = 1
    image[index1, index1:index2] = 1
    image[index2, index1:index2+1] = 1
    return image
#imgplot = plt.imshow(image)
#plt.show()

def create_bin_circle_edges(size, radius):
    image = np.zeros((size, size))
    a, b = int(size/2), int(size/2)
    for y in range(len(image)):
        for x in range(len(image[0])):
            calc = np.sqrt((x - a) ** 2 + (y - b) ** 2)
            if radius - 0.5 < calc < radius + 0.5:
                image[y][x] = 1
    return image


create_bin_circle_edges(21, 5)

#image = create_bin_square_edges(10)
#accumulator, thetas, rhos, lines = hough_line(image, 8)

# Easiest peak finding based on max votes
"""idx = np.argmax(accumulator)
rho = rhos[int(idx / accumulator.shape[1])]
theta = thetas[idx % accumulator.shape[1]]
print("rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta)))"""

"""img = plt.imread('testing-images/greyscale.png')
img = np.array(img)
print(img[:, :, 0])
edges = feature.canny(img[:, :, 0], sigma=3)

lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
                                 line_gap=3)

print(lines)"""


