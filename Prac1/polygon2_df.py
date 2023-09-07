# -*- coding: utf-8 -*-


# It's customary to call pandas pd when importing it
import pandas as pd
polygons = {
    'Name': [
        'Triangle', 'Quadrilateral', 'Pentagon', 'Hexagon', 'Heptagon', 
        'Octagon', 'Nonagon', 'Decagon', 'Hendecagon', 'Dodecagon', 
        'Tridecagon', 'Tetradecagon'],
     # Range parameters are the start, the end of the range and the step
     'Sides': range(3, 15, 1),
}
polygons_data_frame = pd.DataFrame(polygons)

print(polygons_data_frame)

# Sort the df by name and print the first 5 values
head5=polygons_data_frame.sort_values('Name').head(5)
print(head5)

# Method 1: Add the length of name as new column 
polygons_data_frame[
   'Length of Name'
] = polygons_data_frame['Name'].str.len()
#print the updated df
print(polygons_data_frame)

# Method 2: using apply function to create the new column
polygons_data_frame[
   'Length of Name'
] = polygons_data_frame['Name'].apply(len)
print(polygons_data_frame)

# Method 3: using lamda function
polygons_data_frame[
   'Length of Name'
] = polygons_data_frame['Name'].apply(lambda n: len(n))
print(polygons_data_frame)


# We use the DataFrame's plot method here, 
# where we specify that this is a scatter plot
# and also specify which columns to use for x and y
polygons_data_frame.plot(
    title='Sides vs Length of Name',
    kind='scatter',
    x='Sides',
    y='Length of Name',
)




