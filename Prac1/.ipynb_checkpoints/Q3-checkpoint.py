#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 22:59:06 2023

@author: kevinr
"""

# ML Python Basics 

# 3 Write a program which asks users to enter five numbers
# and then prints out the largest if the 5 numbers
  
def get_largest_number(numbers):
    largest = numbers[0]
    for num in numbers:
        if num > largest:
            largest = num
    return largest

def main():
    try:
        numbers = []
        for i in range(5):
            num = float(input(f'Enter number {i+1}: '))
            numbers.append(num)
            
        Lnum = get_largest_number(numbers)
        print(f'The largest number is: {Lnum}')
        
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
        
        
if __name__ == "__main__":
    main()
