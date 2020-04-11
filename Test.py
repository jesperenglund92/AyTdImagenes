import numpy as np

r = np.array(((1, 1, 1),
              (1, 1, 1),
              (1, 1, 1)
              ))
r = np.array(())

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

print(rgb)

#rgb2 = np.dstack((r2, g2, b2))

print(rgb2)

rgb2 = rgb2.reshape(rgb.shape)

#print(rgb2)
