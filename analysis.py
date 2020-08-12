import math , numpy

def nabs(x):
    if x < 0 :
        return -x
    else :
        return x
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
def differentiate_2d(f):
    d_element = .00001
    def wrapper(var,x,y):
        if var == 1 :
            dx = d_element
            return (f(x+dx,y)-f(x-dx,y))/(2*dx)
        if var == 2 :
            dy = d_element
            return (f(x,y+dy)-f(x,y-dy))/(2*dy)
        else:
            raise ValueError("Check parameters in function")
    return wrapper
def differentiate_3d(f):
    d_element = .00001
    def wrapper(var, x, y, z):
        if var == 1:
            dx = d_element
            return (f(x + dx, y , z) - f(x - dx, y,z)) / (2 * dx)
        if var == 2:
            dy = d_element
            return (f(x, y + dy,z) - f(x, y - dy,z)) / (2 * dy)
        if var == 3:
            dz = d_element
            return (f(x,y,z+dz) - f(x,y,z-dz)) / (2 * dz)
        else:
            raise ValueError("Check parameters in function")
    return wrapper
def newton(f,initial_seed,steps = 100,tol=.001):
    x_n = initial_seed
    approx_ar = []
    for i in range(steps):
        x_n_1 = x_n - f(x_n) / differentiate(f)(x_n)
        approx_ar.append(x_n_1)
        if nabs(x_n_1 - x_n ) < tol :
            break
        x_n = x_n_1
    return x_n , approx_ar
def grad_2d(f):
    def wrapper(x,y):
        return [differentiate_2d(f)(1,x,y),differentiate_2d(f)(2,x,y)]
    return wrapper
def grad_3d(f):
    def wrapper(x,y,z):
        return [differentiate_3d(f)(1,x,y,z),differentiate_3d(f)(2,x,y,z),differentiate_3d(f)(3,x,y,z)]
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
class field:
    def __init__(self,fx,fy,fz , grid = None):
        self.fx = fx
        self.fy = fy
        self.fz = fz
        """
        The grid parameter should be given as 3d meshgrid
        """
        self.grid = grid
        if self.grid != None :
            self.X , self.Y , self.Z = self.grid
    def gva(self,x,y,z):
        """
        gva -> get value at (Point)
        :param x: x value
        :param y: y value
        :param z: z value
        :return: value of field as 1x3 array
        """
        return [self.fx(x,y,z),self.fy(y,x,z),self.fz(z,x,y)]
    def div(self,x=None,y=None,z=None):
        """
        :return: the divergence of the field over a given grid
        """
        if self.grid == None and x.all() != None and y.all() != None and z.all() != None:
            return differentiate_3d(self.fx)(1,x,y,z)+differentiate_3d(self.fy)(2,x,y,z)+differentiate_3d(self.fz)(3,x,y,z)+0
        if  self.grid != None:
            div = differentiate_3d(self.fx)(1, self.X, self.Y, self.Z) + differentiate_3d(self.fy)(2, self.X, self.Y, self.Z) + differentiate_3d(self.fz)(3, self.X, self.Y, self.Z) + 0
            return div
        else :
            raise ValueError("Neither a grid , nor a point array  was given")
    def div_at(self,x,y,z):
        if x!=None and y!=None and z!= None:
            return differentiate_3d(self.fx)(1,x,y,z)+differentiate_3d(self.fy)(2,x,y,z)+differentiate_3d(self.fz)(3,x,y,z)+0
        else:
            raise ValueError("No point was given")
    def curl(self,x=None,y=None,z=None):
        """
        :return: the curl of the field over a grid as a 1x3 array
        """
        if self.grid == None and x.all() != None and y.all() != None and z.all != None:
            curl = [differentiate_3d(self.fz)(2, x, y ,z) - differentiate_3d(self.fy)(3,x,y,z)+0,
                    differentiate_3d(self.fz)(1 ,x,y,z) - differentiate_3d(self.fx)(3,x,y,z)+0,
                    differentiate_3d(self.fy)(1,x,y,z) - differentiate_3d(self.fx)(2,x,y,z)+0]
            return curl
        if self.grid != None :
            curl = [differentiate_3d(self.fz)(2, self.X, self.Y, self.Z) - differentiate_3d(self.fy)(3, self.X, self.Y, self.Z),
                differentiate_3d(self.fz)(1, self.X, self.Y, self.Z) - differentiate_3d(self.fx)(3, self.X, self.Y, self.Z),
                differentiate_3d(self.fy)(1, self.X, self.Y, self.Z) - differentiate_3d(self.fx)(2, self.X, self.Y, self.Z)]
            return curl
        else:
            raise ValueError("Neither a grid , nor a specific point was given")
    def curl_at(self,x,y,z):
        if x!=None and y!=None and z!= None:
            curl = [differentiate_3d(self.fz)(2, x, y, z) - differentiate_3d(self.fy)(3, x, y, z) + 0,
                    differentiate_3d(self.fz)(1, x, y, z) - differentiate_3d(self.fx)(3, x, y, z) + 0,
                    differentiate_3d(self.fy)(1, x, y, z) - differentiate_3d(self.fx)(2, x, y, z) + 0]
            return curl
        else:
            raise ValueError("No point was given ")

