

## multi-array
"""

import numpy as np


x = np.arange(3)  # / n = 3
y = np.arange(3)
z = np.arange(3)

multi_array = np.array([x, y, z])

print(multi_array)
print(multi_array.shape)         # (3,3)

"""



## np.linspace
"""

import numpy as np

#w = np.linspace(2, 10, 8)
#print('w = ', w)

b = np.arange(1, 30, 3)
print('b = ', b)

c = np.linspace(1, 30, 3)
print('c = ', c)

d = np.linspace(1, 30, 3, False)
print('d = ', d)

"""



## multi-array indexing
"""

import numpy as np

x = np.arange(1, 4)  # / n = 3
y = np.arange(4, 7)
z = np.arange(7, 10)

multi_array = np.array([x, y, z])

print(multi_array)
print(multi_array.shape)

print(multi_array[0, 2])  # row 0 and col 2
print(multi_array[1, 1])

print(multi_array.dtype)  # int 64

multi_array = np.array([x, y, z], dtype = np.int32)
print(multi_array.dtype)  # int 32


"""



