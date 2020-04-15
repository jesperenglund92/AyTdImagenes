import numpy as np

r = np.array(((1, 2, 3),
              (4, 5, 6),
              (7, 8, 9)
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

rgb = np.dstack((r, g))

rgb = np.dstack((rgb, b))

print(rgb)

def rot(array):
    new_array = array
    for r in range(array.shape[0]):
        for c in range(array.shape[1]):
            check_distance_to_edge(array, (r, c))


def check_distance_to_edge(array, pos):
    pass


rot(r)
