import numpy as np
import matplotlib.pyplot as plt
from cv2 import *
from skimage import data, io, filters, feature
from skimage.transform import probabilistic_hough_line

from PIL import Image

r = np.array(((1, 2, 3, 4),
              (4, 5, 6, 4),
              (7, 8, 9, 4)
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

#rgb = np.dstack((r, g))

#rgb = np.dstack((rgb, b))


def hough_line(img, vote_thresh, theta_step=1, rho_step=1):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, theta_step))
    height, width = img.shape
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


def create_bin_circle_edges(size, radius):
    image = np.zeros((size, size))
    a, b = int(size/2), int(size/2)
    for y in range(len(image)):
        for x in range(len(image[0])):
            calc = np.sqrt((x - a) ** 2 + (y - b) ** 2)
            if radius - 0.5 < calc < radius + 0.5:
                image[y][x] = 1
    return image

def add_circle(image, radius, center):
    a, b = center
    for y in range(len(image)):
        for x in range(len(image[0])):
            calc = np.sqrt((x - a) ** 2 + (y - b) ** 2)
            if radius - 0.5 < calc < radius + 0.5:
                image[y][x] = 1
    return image


def hough_circle(img, thresh, radii):
    # Rho and Theta ranges
    height, width = img.shape
    if radii == None:
        r_max = np.max((height, width))
        r_min = 3
    else:
        r_max, r_min = np.max(radii), np.min(radii)

    R = r_max - r_min + 1  # number of radii to check

    thetas = np.deg2rad(np.arange(0, 360))
    y_indices, x_indices = np.nonzero(img)  # (row, col) indices for borders

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    fin_circles = np.zeros(img.shape)
    for val in range(R):
        r = r_min + val
        # Hough accumulator array
        # accumulator representing r, X and Y respectively
        accumulator = np.zeros((r, width + 2 * r, height + 2 * r))
        #Create circle blueprint
        blueprint = np.zeros((2 * r + 1, 2 * r + 1))
        a, b = r + 1, r + 1  # The center of the blueprint/mask
        for index in range(num_thetas):
            x = int(np.round(r * cos_t[index]))
            y = int(np.round(r * sin_t[index]))
            calc = np.round(np.sqrt(x ** 2 + y ** 2))
            if r - 0.5 < calc < r + 0.5:
                blueprint[a + x - 1, b + y - 1] = 1
        num_points = np.argwhere(blueprint).shape[0]
        # Vote in the hough accumulator
        for i in range(len(x_indices)):
            x = x_indices[i]
            y = y_indices[i]
            X = [x - a + r, x + a + r - 1]  # extreme X values
            Y = [y - b + r, y + b + r - 1]  # extreme Y values
            accumulator[r - 1, X[0]:X[1], Y[0]:Y[1]] += blueprint

        accumulator[r - 1][accumulator[r - 1] < num_points * thresh / r] = 0
        resized_acc = accumulator[:, r:-r, r:-r]
        for r, a, b in np.argwhere(resized_acc):
            r += 1
            a += 1
            b += 1
            for y in range(len(image)):
                for x in range(len(image[0])):
                    calc = np.sqrt((x - a) ** 2 + (y - b) ** 2)
                    if r - 0.5 < calc < r + 0.5:
                        fin_circles[x, y] = 1
    return fin_circles


#image = create_bin_square_edges(10)
#hough = hough_line(image, 8)

image = create_bin_circle_edges(21, 5)
image = add_circle(image, 3, (6, 6))
image[6, 7] = 0
image[7, 6] = 0
image[8, 5] = 0
image[9, 6] = 0
image[9, 7] = 0
image[8, 8] = 0
image[7, 9] = 0
image[6, 9] = 0
image[5, 8] = 0
print(image)
hough = hough_circle(image, 2, (3,5))
print(hough)



