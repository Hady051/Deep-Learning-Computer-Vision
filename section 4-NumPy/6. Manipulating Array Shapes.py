import numpy as np

## ravel() and flatten()
"""

import numpy as np

x = np.arange(9).reshape(3, 3)
print(x)

raveled_array = x.ravel()
print(raveled_array)

flattened_array = x.flatten()
print(flattened_array)

raveled_array[2] = 547841
print('raveled_array = ', raveled_array)
print('x = ', x)

"""



## shape and Transpose
"""


y = np.arange(9)
y.shape = (3, 3)
print(y)

print(y.transpose())

"""



## resizing
"""

y = np.arange(9)
y.shape = (3, 3)
print('y = ', y)

y = np.resize(y, (6, 6) )

print('new y = ', y)

"""



## zeros, ones and eye
"""

print(np.zeros(6, dtype=int))

print(np.zeros((2, 3), dtype=int))

print(np.eye(3) )

"""



## creating arrays with random numbers
##"""

x = np.random.rand(4, 4)
print(x)

#@"""