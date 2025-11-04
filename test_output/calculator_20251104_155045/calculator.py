"""
calculator.py
=============
This module provides a library for basic arithmetic operations including addition, subtraction, multiplication, and division.
"""


def add(a: float, b: float) -> float:
    """
    Returns the sum of two numbers.

    Parameters:
    a (float): The first number
    b (float): The second number

    Returns:
    float: The sum of a and b
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both inputs must be numbers")
    return a + b


def subtract(a: float, b: float) -> float:
    """
    Returns the difference of two numbers.

    Parameters:
    a (float): The first number
    b (float): The second number

    Returns:
    float: The difference of a and b
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both inputs must be numbers")
    return a - b


def multiply(a: float, b: float) -> float:
    """
    Returns the product of two numbers.

    Parameters:
    a (float): The first number
    b (float): The second number

    Returns:
    float: The product of a and b
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both inputs must be numbers")
    return a * b


def divide(a: float, b: float) -> float:
    """
    Returns the quotient of two numbers.

    Parameters:
    a (float): The numerator
    b (float): The denominator

    Returns:
    float: The quotient of a and b

    Raises:
    ZeroDivisionError: If b is zero
    """
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("Both inputs must be numbers")
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b
