#### Python crash course



## New additions to Python
"""

## declaring a variable immediately with print and saving it
print(num:=15)
print(num)

# ** printf method (the newer one)
print(f'this person is {'tall'}, {'slim'} and {'blonde'} )

"""



## Arithmetic Operations
"""

print(1 + 1)

print(1 - 1)

print(7 * 7)

print(10 / 5)

print(6 + 9 * 3 + 3)

print( (6 + 9) * (3 + 3) )

# / modulus operator (remainder)

print(6 % 2)  # / gives 0  (6-2-2-2)

print(9 % 2)  # / gives 1

# / exponent (power)

print(6 ** 2)   # gives 36

# / double-slash: divides and rounds down to the nearest integer

print(13 // 2) # / division gives 6.5 but here it gives 6


"""



## Variables
""" Variables

age = 25
print(age)
name = 'Jane'
gender = 'female'

age, name, gender = 27, 'Rana', 'female'
print(age)
print(name)

# blood type = 'AB' # / this is wrong, as space shouldn't be used to declare variables
blood_type = 'AB'

age = 20
age = age + 5
print(age)

age += 5
print(age)

age -= 5
print(age)

"""



## Numeric Data Types
"""

x = 7
y = 3
z = x / y
print(z)

print(type(x) )  # / integer

x = 7.0
print(type(x) )  # / float

print(type(7/3.5) ) # / float (2.0)

print(type(int(7/3.5) ) ) # / integer (2), because I converted into int

w = int(3.7)  # / gives int 3

float_num = float(3)
print(float_num)  # / gives float 3.0

"""



## String Data Types
"""
name = 'Hady'
print(name)
print(type(name) )

# dialogue = " Ameer said "Hello John", to which John replied"Hello Ameer" "
# / this one is wrong and to fix it, I need to use single quotes for the variable
dialogue = ' Ameer said "Hello John", to which John replied "Hello Ameer" '
print(dialogue)

# dialogue = ' Ameer said "Hello John", to which John replied "you're the best" '
# / this is wrong because you can't use single quote inside double quote.
# / to fix it: use *backslash* to include this quote

dialogue = ' Ameer said "Hello John", to which John replied "you\'re the best" '
print(dialogue)

segment_1 = 'I\'m 25'
segment_2 = 'years old'
full_sentence = segment_1 + ' ' + segment_2 + '. '
print(full_sentence)
long_sentence = full_sentence * 10
print(long_sentence)

# / len function
print(len(long_sentence) )


"""



## Booleans
"""

this_is_cool = True
this_is_not_cool = False

comparison_operation = 1 < 2
print(comparison_operation)

# / *or*: needs only one True statement, to give True, while *and* needs both statements to be True to give True
comparison_operation = 1 < 2 or 2 < 3
print(comparison_operation) # / gives True

comparison_operation = 1 < 2 or 2 > 3
print(comparison_operation)  # / gives True

comparison_operation = 1 < 2 and 2 > 3
print(comparison_operation) # / gives False

comparison_operation = not 1 < 2
print(comparison_operation) # / gives False

"""



## Methods
"""

movie_title = 'Harry Potter and the prisoner of Azkaban'
print(movie_title.upper() )
print(movie_title.lower() )
print(movie_title.count('a') )

"""



## Lists
"""
# / the first element in a list has index 0 and so on
names = ['John', 'Jane', 'Joe']
print(names[0] )

random_variable = [True, False, 'Hello', 1, 1.2]
rv_length = len(random_variable) # / gives 5
print(random_variable[rv_length - 1] ) # / will give error that the index is out of reach


"""



## Slicing
"""
# / the upper index is exclusive, meaning that it doesn't get selected
ordered_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(ordered_numbers[2:10] )

print(ordered_numbers[2:len(ordered_numbers)] )
print(ordered_numbers[ : ] )

print( list(range(0, 100, 2) ) )
print( list(range(0, 100, 5) ) )

show_title = 'Game of Thrones'
print(show_title[2:] )

"""



## Membership Operators
"""
months = ['January', 'February', 'March']
print('June' in months)  # / gives False
print('January' in months)  # / gives True
print('June' not in months)  # / gives True

course = 'python crash course'
print('crash' in course)
print('rash' in course) # / gives True

"""



## Mutability
"""
# / Mutability: something that's liable to change / changeable
# / ** lists are mutable ordered sequence of elements
grocery_list = ['bananas', 'apples', 'cauliflower']
grocery_list[2] = 'rutabagas'

print(grocery_list)

misspelled_vegetable = 'cucomber'
misspelled_vegetable[3] = 'u'      # / this will create an error because strings are immutable

"""



## Mutability II
"""

'''
name = 'Ameer'
other_name = name
name = 'John'
print(name)          # / John
print(other_name)    # / Ameer
'''

books = ['The Catcher in the Rye', 'The Mist', 'Lord of the Rings']
more_books = books
books[0] = 'A song of ice and fire'
print(books)          # / ['A song of ice and fire', 'The Mist', 'Lord of the Rings']
print(more_books)     # / ['A song of ice and fire', 'The Mist', 'Lord of the Rings']

# / which describes mutability and it's importance 

"""



## Functions and methods
"""

## Functions
'''
# ** max function
numbers = [4, 3, 7, 4]
print(len(numbers) )
print(max(numbers) )

names = ['Thomas', 'Gio', 'Zack']
print(max(names) )  # / the max is the string that appears last based on alphabetical order


#names = ['Thomas', 'Gio', 'Zack', 1]
#print(max(names) )  # / here the result can't be done because the data types are not comparable.

# ** min function
print(min(numbers) )
print(min(names) )

# ** sorted function
print(sorted(numbers) )
print(sorted(names) )
'''

## methods
'''
# / methods are different from functions because methods depend on the data type

# ** join method
months = '-'.join(['Jan', 'Feb', 'Mar'] )
print(months)

months = ' '.join(['Jan', 'Feb', 'Mar'] )
print(months)

# ** format method

print('This person is {}, {} and {}'.format('tall', 'slim', 'blond') )

# ** printf method (the newer one)
print(f'this person is {'tall'}, {'slim'} and {'blond'}')

# ** append method / adds an element to the end of a list
months = ['Jan', 'Feb', 'Mar']
months.append('Apr')
print(months)


'''

"""



## Tuples
"""

# / Tuples: are **Immutable ordered sequence of elements
# / use brackets ( ) or parenthesis

traits = ('tall', 'slim', 'blond')
height = traits[0]
build = traits[1]
print(height, build)

#traits[0] = 'short'  # / will give an error


traits = 'tall', 'slim', 'blond'    # / **this is the same as above

# *Tuple unpacking
height, build, hair = traits
print(hair)

height, build, hair = 'tall', 'slim', 'blond'
print(hair)

"""



## Sets
"""

# / sets: **Mutable, **unordered, **unique sequence of elements
# / use curly brackets { }
duplicate_numbers = {1, 2, 1, 2, 3, 3, 5}
unique_numbers = set(duplicate_numbers)   # / to convert to set if it wasn't a set
print(duplicate_numbers)
print(unique_numbers)  # / will output the same thing

unique_numbers.add(4)
print(unique_numbers)

"""



## Dictionaries
"""

# / Dictionary: is composed of key:value, they are **unordered
# / use curly braces { }
# / dictionaries are indexed by their keys

inventory = {'banana': 1.29, 'apples': 2.99, 'papayas': 1.39 }
print(inventory['banana'] )

inventory['banana'] = 2.99
print(inventory)

banana_price = inventory.get('banana')
print(banana_price)

strawberry_price = inventory.get('strawberry')
print(strawberry_price) # / prints none

print('papayas' in inventory)

"""



## Compound Data Structures
"""

grocery_items = {'banana': {'price': 2.99, 'origin': 'Guatemala'},
                 'apple': {'price': 1.29, 'origin': 'UK'},
                 'papayas': {'price': 2.39, 'origin': 'Costa Rica'}
                 }

print(grocery_items['apple'] )

print(grocery_items['apple']['origin'] )

"""



## if, else
"""

grocery_items = {'bananas': 2.99, 'apples':1.29, 'papayas': 2.39}

item = 'brussel sprouts'

if item in grocery_items:
    print('found the ', item)
else:
    print('couldn\'t find the', item)
    grocery_items.update({item: 1.49} )
    print('just added the item, here is the updated grocery list', grocery_items)


"""



## elif
"""

'''
grocery_items = {'bananas': 2.99, 'apples': 1.29, 'papayas': 2.39}

item, price = 'rutabagas', 3.56

if item in grocery_items:
    print('found the ', item)

elif price > 2.99:
    print('too expensive for inventory')

else:
    print('couldn\'t find the', item)
    grocery_items.update({item: price} )
    print('just added the item, here is the updated grocery list', grocery_items)
'''

daytime = 'dawn'

if daytime == 'dawn':
    print('still asleep')
elif daytime == 'morning':
    print('time to go to work')
elif daytime == 'noon':
    print('time to take a lunch break')
elif daytime == 'afternoon':
    print('time to go home')
else:
    print('time to go to sleep')
    

"""



## Complex Comparisons
"""

'''
reynolds_number = 5000

if 2000 < reynolds_number < 10000:
    print('flow regime is transitional')
    
'''

reynolds_number = 1000

if reynolds_number > 2000 and reynolds_number < 10000:
    print('flow is transitional')

if reynolds_number <= 2000 or reynolds_number >= 10000:
    print('flow isn\'t transitional')


"""



## for loops
"""

'''
months = ['Jan', 'Feb', 'Mar']

for month in months:
    print(month)

for num in range(0, 100):
    print(num)

for num in range(0, 100):
    print('hello')
'''

names = ['hillary', 'diana', 'brian']

for index in range(len(names) ):
    names[index] = names[index].title()

print(names)

"""



## for loops II
"""

movies = {'Titanic': 1997, 'Finding Nemo': 2003}

for key in movies:
    print(key)

for key, value in movies.items():
    print(key, value)

for key, value in movies.items():
    print('the movie {}, was made in {}'.format(key, value) )

"""



## while loops
"""

random_number = 20

while random_number <= 30:
    print(random_number)
    random_number += 1

"""



## Break and continue
"""

# / continue: jumps over one iteration of a loop / skips it
# / break: jumps out of the loop completely

numbers = list(range(0, 10) )
print(numbers)

for number in numbers:
    if number % 2 != 0:
        continue
    print(number)

for number in numbers:
    if number % 2 != 0:
        break
    print(number)


"""



## Functions
"""

'''
def num_square(num):
    return print(num * num)

num_square(6)

def rectangle_area(length, width):
    return print('The rectangle area is: ', length * width)

rectangle_area(5, 4)
'''

'''
def BMI_calculator():
    weight = int(input('enter your weight in KG: ') )
    print(f'your weight: {weight} KG')
    height = int(input('enter your height in cm: ') )
    print(f'your height: {height} m')

    return print('your BMI from the givens is: ', (weight / (height ** 2) ) * 10000 )


BMI_calculator()
'''


"""



## Scope
"""

# / local variable: variable names assigned within a function
# / can only be accessed within the block in which they were declared
# / global variable: variable names assigned at the top level of a module file
# / can be accessed in every part of the program

number = 2

def random_function():
    name = 'Bill'
    number = 5

# print(name) # / will give an error
print(number)

"""



## Doc Strings / comment
"""

def rectangle_area(length, width):
    ''' 
    
    INPUT:
    this function takes in two parameters length and width
    
    OUTPUT:
    
    calculates the area of a rectangle based on the length and width provided by the user, 
    where area = length * width 
    
    '''
    return length * width

"""



## Lambda and higher order Functions
"""

numbers = [1, 2, 3, 4, 5, 6]

# ** filter function

#def even_or_odd(number):
#    return number % 2 == 0


#print( list(filter(even_or_odd, numbers ) ) )

# ** lambda operation

print( list(filter(lambda number: number % 2 == 0, numbers ) ) )


"""



