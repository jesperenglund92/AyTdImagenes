import numpy as np


def hough_circles(img, threshold, region, radii=None):
  rows, cols = img.shape
  if radii == None:
    r_max = np.max((rows, cols))
    r_min = 3
  else:
    r_max, r_min = np.max(radii), np.min(radii)

  R = r_max - r_min
  # Initializing accumulator array.
  # Accumulator array is a 3 dimensional array with the dimensions representing
  # the radius, X coordinate and Y coordinate resectively.
  # Also appending a padding of 2 times R_max to overcome the problems of overflow
  A = np.zeros((r_max, rows + 2 * r_max, cols + 2 * r_max))
  B = np.zeros((r_max, rows + 2 * r_max, cols + 2 * r_max))

  # Precomputing all angles to increase the speed of the algorithm
  theta = np.arange(0, 360) * np.pi / 180
  edges = np.argwhere(img[:, :])  # Extract all edge coordinates, argwhere finds indices of elements that are non-zero
  for val in range(R):
    r = r_min + val
    # Creating a Circle Blueprint
    bprint = np.zeros((2 * (r + 1), 2 * (r + 1)))
    m, n = r + 1, r + 1  # Finding out the center of the blueprint
    for angle in theta:
      x = int(np.round(r * np.cos(angle)))
      y = int(np.round(r * np.sin(angle)))
      bprint[m + x, n + y] = 1
    constant = np.argwhere(bprint).shape[0]
    for x, y in edges:  # For each edge coordinates
      # Centering the blueprint circle over the edges
      # and updating the accumulator array
      X = [x - m + r_max, x + m + r_max]  # Computing the extreme X values
      Y = [y - n + r_max, y + n + r_max]  # Computing the extreme Y values
      A[r, X[0]:X[1], Y[0]:Y[1]] += bprint
    A[r][A[r] < threshold * constant / r] = 0

  for r, x, y in np.argwhere(A):
    temp = A[r - region:r + region, x - region:x + region, y - region:y + region]
    try:
      p, a, b = np.unravel_index(np.argmax(temp), temp.shape)
    except:
      continue
    B[r + (p - region), x + (a - region), y + (b - region)] = 1

  return B[:, r_max:-r_max, r_max:-r_max]


def create_bin_circle_edges(size, radius):
  image = np.zeros((size, size))
  a, b = int(size / 2), int(size / 2)
  for y in range(len(image)):
    for x in range(len(image[0])):
      calc = np.sqrt((x - a) ** 2 + (y - b) ** 2)
      if radius - 0.5 < calc < radius + 0.5:
        image[y][x] = 1
  return image

circle = create_bin_circle_edges(21, 5)
hough = hough_circles(circle, 3, 15, radii=5)
for i in hough:
  print(i)
