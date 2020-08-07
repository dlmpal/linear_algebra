import math

def arctangent(x, steps=22, degrees=False):
    # input must be of numerical form
    arctan = 0
    for n in range(steps):
        arctan += (2 ** (2 * n)) * (factorial(n) ** 2) * (x ** (2 * n + 1)) / (
                (factorial(2 * n + 1)) * (1 + x ** 2) ** (n + 1))
    if degrees == True:
        return 57.2957795 * arctan
    else:
        return arctan


def arcsin(x, degrees=False):
    if x > 1: raise ValueError("Input must be between -1 and 1")
    arcsine = 2 * arctangent(x / (1 + (1 - x ** 2) ** (.5)), degrees)
    return arcsine


def arccos(x, degrees=False):
    if x > 1: raise ValueError("Input must be between -1 and 1")
    arccosine = 3.14159265359 / 2 - arcsin(x, degrees)
    return arccosine


def sin(x, steps=22):
    sum = 0
    for n in range(steps):
        sign = -1
        if n % 2 == 0 :
            sign = 1
        sum += sign *  (x ** (2 * n + 1)) / factorial(2 * n + 1)
    return sum


def cos(x, steps=22):
    sum = 0
    for n in range(steps):
        sign = -1
        if n%2 == 0 :
            sign = 1
        sum += sign * (x ** (2 * n)) / (factorial(2 * n))
    return sum


def tan(x, steps=22):
    if cos(x, steps) < 10e-12: raise ValueError("Tangent of multiples of pi/2 is not defiend")
    tangent = sin(x, steps) / cos(x, steps)
    return tangent
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)


def exp(x, steps=20):
    summa = 0
    for i in range(steps):
        summa += x ** i / factorial(i)
    return summa


def differentiate(f):
    """
    differentiates a function using the mean
    :param f: the function to be differentiated
    :return: the value of the derivative at given point
    """
    dx = .00001
    def wrapper(x):
        return ((f(x + dx) - f(x - dx)) / (2 * dx))

    return wrapper


def integrate(f):
    """
    this functions uses a monte carlo approach to integration
    :param f: the function to be integrated
    :return: the value of the integral
    """
    from random import uniform
    def wrapper(a, b, steps):
        integral = 0
        for i in range(steps):
            summ = 0
            for i in range(steps):
                summ += f(uniform(a, b))
            integral += summ
        integral /= 100
        return integral * (b - a) / steps

    return wrapper


def integrate_trapz(f , improper = False):
    """
    integrates a function using the trapezoidal rule
    can be used for improper integrals , given that the bounds are well behaved
    :param f: the function to be integrated
    :param improper: boolen- 1 if bounds are inf
    :return: the value of the integral
    """
    import numpy as np
    if improper == True:
       def g(x):
          return f(1/x) * (1/(x**2))
    else :
        g = f
    def wrapper(a, b, steps):
        x = np.linspace(a, b, steps + 1)
        dx = (b - a) / steps
        y = g(x)
        y_left = y[:-1]
        y_right = y[1:]
        integral = (dx / 2) * np.sum(y_left + y_right)
        return integral
    return wrapper



