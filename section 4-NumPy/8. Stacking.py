
import numpy as np

## h-stack
"""

x = np.arange(4).reshape(2,2)
print('x = \n', x)

y = np.arange(4, 8).reshape(2,2)
print('y = \n', y)


z = np.hstack( (x,y) )
print('z = \n', z)

"""



## v-stack
"""

x = np.arange(4).reshape(2,2)
print('x = \n', x)

y = np.arange(4, 8).reshape(2,2)
print('y = \n', y)


z = np.vstack( (x,y) )
print('z = \n', z)

"""



## concatenate
"""

x = np.arange(4).reshape(2,2)
print('x = \n', x)

y = np.arange(4,8).reshape(2,2)
print('y = \n', y)

z = np.concatenate( (x,y), axis=1)  # horizontal stacking
print('z = \n', z)

w = np.hstack( (x,y) )
print('w = \n', w)

print(z == w)


z = np.concatenate( (x,y), axis=0) # vertical stacking
print('z = \n', z)


"""



## depth stacking
"""

x = np.arange(4).reshape(2,2)
print('x = \n', x)
print(x.shape)

y = np.arange(4,8).reshape(2,2)
print('y = \n', y)
print(y.shape)

depth_stack = np.dstack( (x, y) )
print('depth stack = \n', depth_stack)

print(depth_stack.shape)

"""



## col-stack
"""

x = np.arange(4).reshape(2,2)
print('x = \n', x)
print(x.shape)

y = np.arange(4,8).reshape(2,2)
print('y = \n', y)
print(y.shape)

col_stack = np.column_stack( (x, y) )
print('col stack = \n', col_stack)
print(col_stack.shape)

h_stack = np.hstack( (x,y) )
print(h_stack)

"""



## row_stack
"""

x = np.arange(4).reshape(2,2)
print('x = \n', x)
print(x.shape)

y = np.arange(4,8).reshape(2,2)
print('y = \n', y)
print(y.shape)

row_stack = np.row_stack( (x,y) )   ## depricated (not used anymore)
print('col stack = \n', row_stack)
print(row_stack.shape)

v_stack = np.vstack( (x,y) )
print(v_stack)

"""

