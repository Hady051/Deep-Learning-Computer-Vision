

## reshaping
##"""

import numpy as np

x = np.arange(9)
print(x)

x = x.reshape(3, 3)
print(x)


y = np.arange(18).reshape(2, 3, 3)
print(y)

y = np.arange(18).reshape(3, 3, 2)
print(y)


#@"""