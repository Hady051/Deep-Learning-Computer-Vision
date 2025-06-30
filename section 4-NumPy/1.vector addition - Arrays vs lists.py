

## lists
"""

list_two = list(range(1, 4) ) ** 2

list_three = list(range(1, 4) )

list_sum = []

for index in range(3):   # 0, 1, 2
    list_two[index] = list_two[index] ** 2
    list_three[index] = list_three[index] ** 3
    list_sum.append(list_two[index] + list_three[index] )

print(list_sum)

"""



## numpy arrays
"""

import numpy as np

array_two = np.arange(1, 4) ** 2
print(array_two)

array_three = np.arange(1, 4) ** 3
print(array_three)

print('The sum of the 2 arrays: ', array_two + array_three)


"""



## numpy math functions
"""

import numpy as np


pow_array = np.power(np.array([1, 2, 3]), 4 )
print(pow_array)

neg_aeray = np.negative(np.array([2, 4, 6, -5]) )
print(neg_aeray)

sample_array = np.array([2, 4, 6])

exp_array = np.exp(sample_array)
print(exp_array)

log_array = np.log(sample_array)
print(log_array)

sin_array = np.sin(sample_array)
print(sin_array)


"""