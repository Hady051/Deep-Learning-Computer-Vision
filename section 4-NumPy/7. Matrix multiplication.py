

## Matrix multiplication
##"""

import numpy as np

mat_a = np.matrix([0, 3, 5, 5, 5, 2]).reshape(2,3)
print('mat A = \n', mat_a)
mat_b = np.matrix([3, 4, 3, -2, 4, -2]).reshape(3, 2)
print('mat B = \n', mat_b)

mat_ab = mat_a * mat_b
print('mat AB = \n', mat_ab)

mat_AB = np.matmul(mat_a, mat_b)
print('mat_AB = \n', mat_AB)

print(mat_a @ mat_b)

#@"""