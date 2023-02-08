#Melvin Moreno
#COMP6710
#Workshop 1
#Date: 1/25/2023
# A simple calculator that multiplies and divides.
# It also handles division by zero exceptions.

def performMul(a, b):
    return a * b

def performDiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return "Division by zero is not allowed"