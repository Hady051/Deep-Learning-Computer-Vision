#### TODO 1: Importing the Libraries
##"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D

import random

## New
import pickle  # // to unpickle pickled files
import pandas as pd

#@"""



#### TODO 2: Loading/Opening the files
##"""

np.random.seed(0)

with open('traffic-signs/train.p', 'rb') as file:
    train_data_file = pickle.load(file)

with open('traffic-signs/test.p', 'rb') as file:
    test_data_file = pickle.load(file)

with open("traffic-signs/valid.p", "rb") as file:
    validation_data_file = pickle.load(file)

print(type(train_data_file) )  # -> <class 'dict'>

## X_train, y_train, ...
train_images, train_labels = train_data_file["features"], train_data_file["labels"]
test_images, test_labels = test_data_file["features"], test_data_file["labels"]
validation_images, validation_labels = validation_data_file["features"], validation_data_file["labels"]


### Des:
## Line: with
'''
## with
# // is used whenever you wish to execute two operations as a pair and have a block of code in between
'''

#@"""



#### TODO 3: Dataset Parameters
##"""

# print(train_images.shape)
# print(test_images.shape)
# print(validation_images.shape)
# O
'''
(34799, 32, 32, 3)
(12630, 32, 32, 3)
(4410, 32, 32, 3)
'''

# print(train_labels.shape)
# O
'''
(34799,)    # // this means (34799, 1)
'''

### Des:
'''
# // for the train images, we only have 34799 images which are 32x32 pixels and rgb (3D).
# // same description for the rest with different values.

'''

#@"""



#### TODO 4: assert method
##"""

assert (train_images.shape[0] == train_labels.shape[0] ), "The number of images is not equal to the number of labels"
assert (validation_images.shape[0] == validation_labels.shape[0]), ("The number of images is not equal "
                                                                    "to the number of labels")
assert (test_images.shape[0] == test_labels.shape[0]), "The number of images is not equal to the number of labels"

assert (train_images.shape[1:] == (32, 32, 3) ), "The dimensions of the images are not 32 x 32 x 3"
assert (validation_images.shape[1:] == (32, 32, 3) ), "The dimensions of the images are not 32 x 32 x 3"
assert (test_images.shape[1:] == (32, 32, 3) ), "The dimensions of the images are not 32 x 32 x 3"

#@"""



#### TODO 5: Using Pandas Library to read the csv file
##"""

dataset = pd.read_csv('traffic-signs/signnames.csv')
# print("Dataset = \n", dataset)
# O
'''
Dataset = 
     ClassId                                           SignName
0         0                               Speed limit (20km/h)
1         1                               Speed limit (30km/h)
2         2                               Speed limit (50km/h)
3         3                               Speed limit (60km/h)
4         4                               Speed limit (70km/h)
5         5                               Speed limit (80km/h)
6         6                        End of speed limit (80km/h)
7         7                              Speed limit (100km/h)
8         8                              Speed limit (120km/h)
9         9                                         No passing
10       10       No passing for vechiles over 3.5 metric tons
11       11              Right-of-way at the next intersection
12       12                                      Priority road
13       13                                              Yield
14       14                                               Stop
15       15                                        No vechiles
16       16           Vechiles over 3.5 metric tons prohibited
17       17                                           No entry
18       18                                    General caution
19       19                        Dangerous curve to the left
20       20                       Dangerous curve to the right
21       21                                       Double curve
22       22                                         Bumpy road
23       23                                      Slippery road
24       24                          Road narrows on the right
25       25                                          Road work
26       26                                    Traffic signals
27       27                                        Pedestrians
28       28                                  Children crossing
29       29                                  Bicycles crossing
30       30                                 Beware of ice/snow
31       31                              Wild animals crossing
32       32                End of all speed and passing limits
33       33                                   Turn right ahead
34       34                                    Turn left ahead
35       35                                         Ahead only
36       36                               Go straight or right
37       37                                Go straight or left
38       38                                         Keep right
39       39                                          Keep left
40       40                               Roundabout mandatory
41       41                                  End of no passing
42       42  End of no passing by vechiles over 3.5 metric ...
'''

#@"""



#### TODO 6: Plotting Samples of the Traffic data (from (10.1) MNIST for clarification) # // doesn't work on Pycharm
##"""

number_of_samples = []
number_of_classes = 43
cols = 5

fig, axes = plt.subplots(nrows=number_of_classes, ncols=cols, figsize=(5, 50) )
# // figsize is the size of each grid (subplot), first the width then the height

fig.tight_layout()
# // this is used to automatically deal with overlapping and separates the figures/subplots better

for col in range(cols): # // each col of the 5
    for r_index, row in dataset.iterrows(): # // each row(class) of the 10
        image_selected = train_images[train_labels == r_index]

        axes[r_index][col].imshow(image_selected[random.randint(0, len(image_selected) - 1), : , : ],
                          cmap=plt.get_cmap("gray") )
        axes[r_index][col].axis("off")

        if col == 2:
            axes[r_index][col].set_title("Index: " + str(r_index) + "-" + row["SignName"] )

            number_of_samples.append(len(image_selected) )


### Des:
## Line 176: dataset.iterrows()
'''
# // instead of iterating through range like MNIST, we're iterating through this dataset.iterrows()

## dataset.iterrows():
# // This is a pandas property, this allows us to iterate through our dataframe rows as index and series pairs.
[index, Series]

# // index is like iterating through the rows like before, index corresponds to the Class ID as well.
# // Series is a 1D labeled array which holds the relevant data for each row 
# / like in this dataset -> Class ID, SignName   ex: for index:0 , series = [0, Speed limit (20km/h) ]

'''

#@"""


#### I stopped continuing here
















