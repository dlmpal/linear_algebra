# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 00:54:38 2020

@author: User

This is a mathematical library designed for computations in engineering and scientific applications. 
It includes linear algebra operations and algorithms , as well as basic complex number operations and 
some analytic functions. 

#####
Most functions defined in the script make use of the special classes built inside the module. These are the vector 
,the matrix and the complex number written as complex_num. The vector is constructed from  a 1d python list , while the matrix 
is a 1d list of vectors. Complex numbers are vectors containing two elements , the first of which is the real part , while the latter
the imaginary. 
####

This module was created by JP.


"""

class vector():
    def __init__(self,array):
        self.array = array 
        if self.array == None :
            self.array = []
            
        else:
            self.array = array 
        self.Length = len(self.array)
        
    def zeroes(self,length):
        zero_ar = []
        for i in range(length):
            zero_ar.append(0)
        self.array = zero_ar    
    def printf(self):
        print(self.array)
    def transpose(self):
        length= len(self.array)
        T = [[i]for i in range(length)]


def factorial(n):
    if n == 0:
        return 1
    return  n * factorial(n-1)


def arctangent(x,steps = 22,degrees = False):
    #input must be of numerical form 
    arctan = 0 
    for n in range(steps):
        arctan += (2**(2*n))*(factorial(n)**2)*(x**(2*n + 1))/((factorial(2*n + 1))*(1+x**2)**(n+1))
    if degrees == True : 
        return 57.2957795*arctan
    else: 
        return arctan 
def arcsin(x, degrees = False):
    if x > 1 : raise ValueError("Input must be between -1 and 1")
    arcsine = 2*  arctangent(x/(1 + (1- x**2)**(.5)) ,degrees )
    return arcsine

def arccos(x , degrees = False):
    if x > 1 : raise ValueError("Input must be between -1 and 1")
    arccosine = 3.14159265359/2 - arcsin(x,degrees)
    return arccosine

        
def sin(x,steps = 22):
    sum =0 
    for n in range(steps):
        sum += (-1**n) * (x**(2*n +1)) / factorial(2*n + 1)
    return sum 
    
def cos(x, steps =22 ):
    sum = 0 
    for n in range(steps):
        sum += (-1**n) * (x**(2*n)) / (factorial(2*n))
    return sum    

def tan(x , steps = 22):
    if  cos(x,steps) < 10e-12 : raise ValueError("Tangent of multiples of pi/2 is not defiend")
    tangent = sin(x,steps) / cos(x,steps)
    return tangent
    
        
    
    

class complex_num():
    def __init__(self,a_vector):
        self.a_vector = a_vector 
        if  type(self.a_vector) != vector or self.a_vector.Length != 2 :
            raise TypeError("Input must be a 2d vector , where the first element is the real part and the second the imaginary")
        self.real = a_vector.array[0]
        self.im = a_vector.array[1]
        self.mag = ((self.a_vector.array[1])**2 + (self.a_vector.array[0])**2 )**(.5)
        self.inv = self.a_vector.array[1]/self.a_vector.array[0]
        self.angle = arctangent(self.inv,degrees = True)                  
    def get_angle(self,show = False):
        self.inv = self.a_vector.array[1]/self.a_vector.array[0]
        self.angle = arctangent(self.inv,degrees = True)
        if show == True : print("angle in degrees is ",self.angle)
        return self.angle
    def get_real(self):
        return self.real
    def get_im(self):
        return self.im
    def printf(self):
        print(self.real,"+ j",self.im)
        return 1
    def to_polar(self,show = False):
        magnitude = self.get_mag()
        angle = self.get_angle()
        if show == True : print(magnitude,"<",angle)
        return [magnitude,angle]


def add_complex(c1,c2):
     if type(c1) != complex_num or type(c2) != complex_num:
        raise TypeError("Inputs must be both complex numbers")
     addition_vector = vector([c1.real+c2.real,c1.im+c2.im])
     c3 = complex_num(addition_vector)
     return c3

def mult_complex(c1,c2):
    if type(c1) != complex_num or type(c2) != complex_num:
        raise TypeError("Both inputs must be complex numbers")
    mult_vector = vector([c1.real * c2.real - c1.im*c2.im,c1.im*c2.real + c1.real*c2.im])  
    c3 = complex_num(mult_vector)
    return c3

def to_cartesian(magnitude,angle):
    _REAL = magnitude * cos(angle)
    _IM = magnitude * sin(angle)
    vec = vector([_REAL,_IM])
    c3 = complex_num(vec)
    return c3
    
    

def divide_complex(c1,c2):
    if type(c1) != complex_num or type(c2) != complex_num:
        raise TypeError("Both inputs must be complex numbers")
    mag1 = c1.mag    
    mag2 = c2.mag
    ag1 = c1.angle
    ag2 = c2.angle
    new_magnitude = mag1 / mag2 
    new_angle = ag1 / ag2
       
    return to_cartesian(new_magnitude,new_angle)
             

#input for dot product must be of vector class // all inputs must be dimensionally EQUAL!!!
def entry_prod(*argv):     
    count = 1
    
    for vec in argv : 
        if type(vec) != vector:
            raise TypeError("Input must be of vector class")
        current_len = vec.Length
        if current_len == 0 :
            raise ValueError("length must be 1 or above")
        if count == 1 :
            count +=1 
            common_len = current_len
        elif count != 1 and common_len != current_len :
            raise ValueError("Vectors must be of same length")
    product_ar = [1 for i in range(common_len)]       
    for vec in argv : 
        for entry in range(vec.Length):
            product_ar[entry] *= vec.array[entry] 
    return product_ar        

def dot_product(*argv):
    result = entry_prod(*argv)
    return sum(result)

def cross_product3d(vec1,vec2):
    if type(vec1) != vector or type(vec2) != vector:
        raise TypeError("Inputs must be vectors")
    if vec1.Length != vec2.Length or vec1.Length !=3 :
        raise ValueError("Vector length must be 3")
    #cross_vector = [1 for i in range(vec1.Length)] 
    cross_vector = vector([vec1.array[1]*vec2.array[2] - vec1.array[2]*vec2.array[1],
                    -vec1.array[0]*vec2.array[2]+vec1.array[2]*vec2.array[0],
                    vec1.array[0]*vec2.array[1] -vec1.array[1]*vec2.array[0]
                    ])
    return cross_vector    
        
             
        
            
            
        



def testfunc(*argv):
    count = 1
    for vec in argv:
        if type(vec) != vector:
            raise TypeError("Input must be of vector class")
        current_len = vec.Length
        if count == 1 :
            count +=1 
            common_len = current_len
        elif count != 1 and common_len != current_len :
            raise ValueError("Vectors must be of same length")
    

class matrix():
    def __init__(self,rows,cols):
        self.rows = rows
        self.cols = cols
        self.matrixx = []
        self.matrixt = [list() for i in range(cols)]
        if type(self.rows) != int or type(self.cols) != int:
            raise TypeError("Invalid dimensions")
    def construct(self,*argv):
       testfunc(*argv)
       count = 0 
       
       for vec in argv:
           count += 1 
           if vec.Length != self.cols: 
               raise ValueError("Unconstructable : check length")
           if count > self.rows :
               raise ValueError("matrix of",str(self.rows),"rows cannot fit current number of rows")
           self.matrixx.append(vec.array)
       count = 0 
       for vec in argv :
           for i in range(vec.Length):
               self.matrixt[i].append(vec.array[i])
           count +=1    
               
           
    def printf(self):
        for i in self.matrixx :
            print(i)
    def printfT(self):
        print(self.matrixt)
        
        
def matrix_mult(matrix1,matrix2):
   if matrix1.cols != matrix2.rows:
         raise ValueError("Not compatible matrices")
   matrix3 = matrix(matrix1.rows,matrix2.cols)
   
   
   for i in range(matrix1.rows):
       current_vec = [dot_product(vector(matrix1.matrixx[i]),vector(matrix2.matrixt[j])) for j in range(matrix2.cols)]
       matrix3.matrixx.append(current_vec)
       
   return matrix3
   

def matrix_add(matrix1 , matrix2):
    array1 = matrix1.matrixx
    array2 = matrix2.matrixx
    if type(matrix1) != matrix or type(matrix2) != matrix :
        raise TypeError("Inputs must be matrices")
    if matrix2.cols != matrix1.cols or matrix1.rows != matrix2.rows:
        raise ValueError("Dimensions not combatible")
    
    matrix3 = matrix(matrix1.rows,matrix1.cols)
    for j in range(matrix1.rows):
        for i in range(matrix1.cols):
            array3 = []
            array3.append(array1[j][i]+array2[j][i])
        matrix.matrixx.append(array3)    
    return matrix3

def determinant_3d(a_matrix):
    
    vec1 = vector(a_matrix.matrixx[1])
    vec2 = vector(a_matrix.matrixx[2])
    vec3 = a_matrix.matrixx[0]
    determinant = [(vec1.array[1]*vec2.array[2] - vec1.array[2]*vec2.array[1])*vec3[0],
                   ( -vec1.array[0]*vec2.array[2]+vec1.array[2]*vec2.array[0])*vec3[1],
                    (vec1.array[0]*vec2.array[1] -vec1.array[1]*vec2.array[0])*vec3[2]
                    ]
    sum = 0 
    for i in range(len(determinant)): sum+= determinant[i]
    return sum



def determinant(a_matrix,d=0,show_current_det = False):
    
    if type(a_matrix) != matrix or a_matrix.rows != a_matrix.cols:
        raise TypeError("Input must be square nxn matrix")
    
    
    if a_matrix.rows == 1 :
        return 0 
    if a_matrix.rows == 2 : 
        return a_matrix.matrixx[0][0] * a_matrix.matrixx[1][1] - a_matrix.matrixx[0][1]*a_matrix.matrixx[1][0]
    
    else:
        sub_array = []
        new_matrix = matrix(a_matrix.rows-1,a_matrix.cols-1)        
        #print("LINE301:",sub_array)
        for i in range(a_matrix.cols):
            new_matrix = matrix(a_matrix.rows-1,a_matrix.cols-1)
            new_matrix.matrixx = []
            for j in range(1,a_matrix.rows):
                
               # print("j is:",j)
            
                for k in range(i):
                       sub_array.append(a_matrix.matrixx[j][k])
                       
                #print(sub_array)     
                for k in range(i+1 , a_matrix.cols): 
                    sub_array.append(a_matrix.matrixx[j][k])
                 #   print(sub_array)
                new_matrix.matrixx.append(sub_array) 
                sub_array = []
            #print(sub_array)        
            
            #new_matrix.printf()
            if i % 2 == 0 : 
                sign = 1
            else: sign = -1 
            #print("error:",d)
            d += sign*a_matrix.matrixx[0][i] * determinant(new_matrix)
            if show_current_det == True : print("current det is:",d)
    
        return d     
                    
                    
def rref(A_matrix):
    
    #check for input compatability 
    if type(A_matrix) != matrix or A_matrix.rows != A_matrix.cols:
        raise TypeError("Input must be n x n matrix")
    
    a_matrix = A_matrix.matrixx #obtain array like object from matrix type object
    
    n = A_matrix.rows 
    #gaussian elimination
    for k in range(n-1):
        #row swapping if necessary
        #actual elimination 
        for i in range(k+1,n):
            if a_matrix[i][k] == 0 : continue
            factor = a_matrix[k][k] / a_matrix[i][k] 
            for j in range(k,n):
                a_matrix[i][j] = a_matrix[k][j] - a_matrix[i][j]*factor
    return(a_matrix)                       
    
        
        
        
                                                                                                               
    
def nabs(num):
    if num < 0 :
        return -num
    else :
        return num
           
            
           
class real_func:
    def __init__(self,x):
        self.variable = x
        
def gauss(A_matrix,b_vector,print_rref=False,show = False):
    
    #check for input compatability 
    if type(A_matrix) != matrix or A_matrix.rows != A_matrix.cols:
        raise TypeError("Input must be n x n matrix")
    if type(b_vector) != vector   or b_vector.Length != A_matrix.rows:
        raise TypeError("Constant vector must be a vector of length equal to the size of the matrix")
    a_matrix = A_matrix.matrixx #obtain array like object from matrix type object
    b = b_vector.array #obtain array like object from vector type object
    n = A_matrix.rows 
    #gaussian elimination
    for k in range(n-1):
        #row swapping if necessary
# =============================================================================
#         if nabs(a_matrix[k][k]) < (10**-12):
#             for i in range(k+1,n):
#                 if nabs(a_matrix[i][k]) > nabs(a_matrix[k][k]):
#                     a_matrix[k][i] , a_matrix[i][k] = a_matrix[i][k] , a_matrix[k][i]
#                     b[i] , b[k] = b[k] , b[i]
# =============================================================================
        #actual elimination 
        for i in range(k+1,n):
            if a_matrix[i][k] == 0 : continue
            factor = a_matrix[k][k] / a_matrix[i][k] 
            for j in range(k,n):
                a_matrix[i][j] = a_matrix[k][j] - a_matrix[i][j]*factor
            b[i] = b[k] - b[i]*factor
            
    x = [0 for i in range(n)] #initializing solution vector       
    if print_rref == True :
        print(a_matrix) #prints reduced row echelon form of parameter matrix
        print(b)        #################################################
    
    #check for existence of solutions
    for i in range(n-1 , 0 , -1):
        if a_matrix[i][n-1] == 0 and b[i] != 0 : print("no solutions exist"); return(1)
    
    # backward substition to compute solutions
    x[n-1] = b[n-1]/a_matrix[n-1][n-1] 
    for i in range(n-2,-1,-1):
        sum_ax = 0
        for j in range(i+1,n):
            sum_ax += a_matrix[i][j] * x[j]
        x[i] = (b[i] - sum_ax)/ a_matrix[i][i]
        #print(x[i])
        
    if show == True:
           print("the solution of the linear system is\n",x)
    
    return x
        
        
        
def raise_matrix(a_matrix,power):
    if type(a_matrix) != matrix or a_matrix.cols != a_matrix.rows:
        raise TypeError("Input must be a square nxn matrix")
    raised_matrix = a_matrix    
    for i in range(power):
        raised_matrix = matrix_mult(raised_matrix,a_matrix)
    return raised_matrix    
        


     
def  matrix_by_scalar(a_matrix,scalar):
    if type(a_matrix) != matrix :
        raise TypeError("Input must be a matrix")
    for i in range(a_matrix.rows):
        for j in range(a_matrix.cols):
            a_matrix.matrixx[i][j] *= scalar
    return a_matrix        
        
    


def trace(a_matrix):
    if type(a_matrix) != matrix or a_matrix.rows != a_matrix.cols:
        raise TypeError("Input must be sqaure matrix")
    tr = 0 
    for i in range(a_matrix.cols):
        tr += a_matrix.matrixx[i][i]
    return tr    
def eigen_3d(a_matrix):
#this function is optimized for commonly found 3x3 matrices and can only be used for these dimensions
    if type(a_matrix) != matrix or a_matrix.rows() != 3 or a_matrix.cols() != 3 : 
        raise TypeError("Input must be a 3x3 matrix")
    lambda_matrix = matrix(3,3)
    determinant = determinant_3d(a_matrix)


def _check_for_symmetry(a_matrix,stop = False):
    if type(a_matrix) != matrix or a_matrix.rows != a_matrix.cols : 
        raise TypeError("Input must be a square nxn matrix")
    array = a_matrix.matrixx
    for  r in range(a_matrix.rows):
        for c in range(a_matrix.cols):
            if r == c: continue
            if array[r][c] == array[c][r]:continue
            if stop == True : raise ValueError("Matrix not symmetric")
            return False
    return True        

def normalise_matrix(a_matrix):
    if type(a_matrix) != matrix : 
        raise TypeError("Input must be a square nxn matrix")
    det = determinant(a_matrix)
    if det == 0 : raise ValueError("Matrix cannot be normalised")
    normalised_matrix = matrix_by_scalar(a_matrix, 1/det)
    return normalised_matrix
####check this one ######################################################
def normalised_vector(a_vector):
    if type(a_vector) != vector:
        raise TypeError("Input must be a vector")
    sum = 0     
    for i in range(a_vector.Length):
        sum += (a_vector.array[i])**2
    norm = sum**(0.5)
    for i in range(a_vector.Length):
        a_vector.array[i] *= 1/norm
    return a_vector    
        

###########finish code here
##################################################    
def LU_decomp(a_matrix):
        return 1
        
    
        

def _main():
     if __name__ == "__main__":
         print("cuurently in module")
     else:
         print("Using linear algebra module")
_main()         
        
        




                   
        
           
            
        
        
        
        