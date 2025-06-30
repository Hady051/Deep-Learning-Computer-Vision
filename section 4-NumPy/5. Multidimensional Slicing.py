

## multidim slicing
"""

import numpy as np

#x = np.arange(18).reshape(3, 2, 3)
#print(x)
#print(x[2, 1])

x = np.arange(18).reshape(3, 2, 3)

print(x)

# / indexing
print(x[1, 1, 1])

# / slicing
print(x[1, 0:2, 0:3])

print(x[1, 0:, 0:])

print(x[1, :, :])

print(x[ : , 0, 0] )

print(x[ : , : , :  :2])


"""



## conditional selection
##"""

import numpy as np

x = np.arange(18).reshape(3, 2, 3)

comparison_operation = x > 5

print(comparison_operation)

print(x[comparison_operation])

print(x[x > 5])


#@"""