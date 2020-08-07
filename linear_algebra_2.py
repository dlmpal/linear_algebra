class vector():
    def __init__(self, array):
        self.array = array
        if self.array == None:
            self.array = []

        else:
            self.array = array
        self.Length = len(self.array)

    def zeroes(self, length):
        zero_ar = []
        for i in range(length):
            zero_ar.append(0)
        self.array = zero_ar

    def printf(self):
        print(self.array)

    def transpose(self):
        length = len(self.array)
        T = [[i] for i in range(length)]




from analysis import sin , cos , arccos , arcsin , arctangent , tan , factorial

class complex_num():
    def __init__(self, comps,get_angle = 0 ):
        if type(comps) == vector :
            self.a_vector = comps
        elif type(comps) == list :
            self.comps = comps
            self.a_vector = vector(comps)
        if type(self.a_vector) != vector or self.a_vector.Length != 2:
             raise TypeError(
                 "Input must be a 2d vector , where the first element is the real part and the second the imaginary")
        self.real = self.a_vector.array[0]
        self.im = self.a_vector.array[1]
        if get_angle == 1 :
             self.mag = ((self.a_vector.array[1]) ** 2 + (self.a_vector.array[0]) ** 2) ** (.5)
             if self.real == 0 :
                  self.inv = None
                  self.angle = 90
             else:
                  self.inv = self.a_vector.array[1] / self.a_vector.array[0]
                  self.angle = arctangent(self.inv, degrees=True)

    def get_real(self):
        return self.real

    def get_im(self):
        return self.im

    def printf(self):
        print(self.real, "+ j", self.im)
        return 1

    def to_polar(self, show=False):
        magnitude = self.mag
        angle = self.angle
        if show == True: print(magnitude, "<", angle)
        return [magnitude, angle]


def add_complex(c1, c2):
    if type(c1) != complex_num or type(c2) != complex_num:
        raise TypeError("Inputs must be both complex numbers")
    addition_vector = vector([c1.real + c2.real, c1.im + c2.im])
    c3 = complex_num(addition_vector)
    return c3

def mult_complex(c1, c2):
    if type(c1) != complex_num or type(c2) != complex_num:
        raise TypeError("Both inputs must be complex numbers")
    mult_vector = vector([c1.real * c2.real - c1.im * c2.im, c1.im * c2.real + c1.real * c2.im])
    c3 = complex_num(mult_vector)
    return c3


def to_cartesian(magnitude, angle):
    _REAL = magnitude * cos(angle)
    _IM = magnitude * sin(angle)
    vec = vector([_REAL, _IM])
    c3 = complex_num(vec)
    return c3


def divide_complex(c1, c2):
    if type(c1) != complex_num or type(c2) != complex_num:
        raise TypeError("Both inputs must be complex numbers")
    mag1 = c1.mag
    mag2 = c2.mag
    ag1 = c1.angle
    ag2 = c2.angle
    new_magnitude = mag1 / mag2
    new_angle = ag1 / ag2

    return to_cartesian(new_magnitude, new_angle)

import math
def exp_complex(theta):
    """
    Complex exponent function :
    exp(i*theta) = cos(theta) + i*sin(theta)
    :param theta: theta
    :return: complex number type -> vector([cos(theta),sin(theta)])
    """
    return complex_num(vector([math.cos(theta),math.sin(theta)]))

def raise_complex(c_num,power):
    if type(c_num) != complex_num or power < 0 :
        raise ValueError("Input must be a complex number , power must be positive")
    current = c_num
    for i in range(power-1):
        current = mult_complex(current , c_num)
    return current


# input for dot product must be of vector class // all inputs must be dimensionally EQUAL!!!
def entry_prod(*argv):
    count = 1

    for vec in argv:
        if type(vec) != vector:
            raise TypeError("Input must be of vector class")
        current_len = vec.Length
        if current_len == 0:
            raise ValueError("length must be 1 or above")
        if count == 1:
            count += 1
            common_len = current_len
        elif count != 1 and common_len != current_len:
            raise ValueError("Vectors must be of same length")
    product_ar = [1 for i in range(common_len)]
    for vec in argv:
        for entry in range(vec.Length):
            product_ar[entry] *= vec.array[entry]
    return product_ar


def dot_product(*argv):
    result = entry_prod(*argv)
    return sum(result)


def cross_product3d(vec1, vec2):
    if type(vec1) != vector or type(vec2) != vector:
        raise TypeError("Inputs must be vectors")
    if vec1.Length != vec2.Length or vec1.Length != 3:
        raise ValueError("Vector length must be 3")
    # cross_vector = [1 for i in range(vec1.Length)]
    cross_vector = vector([vec1.array[1] * vec2.array[2] - vec1.array[2] * vec2.array[1],
                           -vec1.array[0] * vec2.array[2] + vec1.array[2] * vec2.array[0],
                           vec1.array[0] * vec2.array[1] - vec1.array[1] * vec2.array[0]
                           ])
    return cross_vector


def testfunc(*argv):
    count = 1
    for vec in argv:
        if type(vec) != vector:
            raise TypeError("Input must be of vector class")
        current_len = vec.Length
        if count == 1:
            count += 1
            common_len = current_len
        elif count != 1 and common_len != current_len:
            raise ValueError("Vectors must be of same length")


class matrix():
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.matrixx = []
        self.matrixt = [list() for i in range(cols)]
        if type(self.rows) != int or type(self.cols) != int:
            raise TypeError("Invalid dimensions")
    def construct(self, *argv):
        testfunc(*argv)
        count = 0

        for vec in argv:
            count += 1
            if vec.Length != self.cols:
                raise ValueError("Unconstructable : check length")
            if count > self.rows:
                raise ValueError("matrix of", str(self.rows), "rows cannot fit current number of rows")
            self.matrixx.append(vec.array)
        count = 0
        for vec in argv:
            for i in range(vec.Length):
                self.matrixt[i].append(vec.array[i])
            count += 1

    def complex_print(self):
        for i in range(self.rows):
            line = []
            for j in range(self.cols):
                 real = self.matrixx[i][j].real
                 im = self.matrixx[i][j].im
                 c_num = str(real)+" + "+"j"+str(im)
                 line.append(c_num)
            print(line)
            line = []



    def summ(self, row_start, row_end, col_start, col_end):
        total = 0
        if row_start == self.rows: row_start -= 1
        if row_end == self.rows: row_end -= 1
        if col_start == self.cols: col_start -= 1
        if col_end == self.cols: col_end -= 1

        if (row_start > row_end or col_start > col_end):
            raise ValueError("Slicing starts must be greater than slicing ")
        if row_start == row_end and col_start != col_end:
            for c in range(col_start, col_end + 1): total += self.matrixx[row_start][c]
            return total
        if col_start == col_end and row_start != row_end:
            for r in range(row_start, row_end + 1): total += self.matrixx[col_start][r]
            return total
        if col_start == col_end and row_start == row_end: return self.matrixx[row_start][col_end]
        for r in range(row_start, row_end + 1):
            for c in range(col_start, col_end + 1):
                total += self.matrixx[r][c]
        return total


    def printf(self):
        for i in self.matrixx:
            print(i)

    def printfT(self):
        print(self.matrixt)
    def shape(self):
        print(self.rows,"x",self.cols)

    def zeroes(self):
        sub_ar = [0 for i in range(self.cols)]
        self.matrixx = [sub_ar for i in range(self.rows)]
    def ones(self):
        sub_ar = [ 1 for i in range(self.cols)]
        self.matrixx = [sub_ar for i in range(self.rows)]

def transpose(a_matrix):
    if type(a_matrix) != matrix :
        raise TypeError("Input must be a matrix")
    matrix2 = matrix(a_matrix.cols , a_matrix.rows)
    ar = []
    for i in range( a_matrix.cols):
        for j in range(a_matrix.rows):
            ar.append(a_matrix.matrixx[j][i])
        matrix2.matrixx.append(ar)
        ar = []

    return matrix2



def matrix_mult(matrix1, matrix2):
    if type(matrix1) != matrix or type(matrix2) != matrix:
        raise TypeError("Both inputs must be matrices")
    if matrix1.cols != matrix2.rows:
        raise ValueError("Incopatible matrices:",matrix1.rows,"x",matrix1.cols," * " , matrix2.rows,"x",matrix2.cols)
    sub = []
    ar3 = []
    row_sum = 0
    for i in range(matrix1.rows):
        for j in range(matrix2.cols):
            for k in range(matrix2.rows):
                row_sum += matrix1.matrixx[i][k] * matrix2.matrixx[k][j]
            sub.append(row_sum)
            row_sum = 0
        ar3.append(sub)
        sub = []
    matrix3 = matrix(matrix1.rows, matrix2.cols)
    matrix3.matrixx = ar3

    return matrix3
def matrix_mult_complex(matrix1,matrix2):
    if type(matrix1) != matrix or type(matrix2) != matrix:
        raise TypeError("Both inputs must be matrices")
    if matrix1.cols != matrix2.rows:
        raise ValueError("Incopatible matrices:",matrix1.rows,"x",matrix1.cols," * " , matrix2.rows,"x",matrix2.cols)
    sub = []
    ar3 = []
    row_sum = complex_num(vector([0,0]))
    for i in range(matrix1.rows):
        for j in range(matrix2.cols):
            for k in range(matrix2.rows):
                row_sum =  add_complex(row_sum,mult_complex(matrix1.matrixx[i][k] , matrix2.matrixx[k][j]))

            sub.append(row_sum)
            row_sum = complex_num(vector([0,0]))
        ar3.append(sub)
        sub = []
    matrix3 = matrix(matrix1.rows, matrix2.cols)
    matrix3.matrixx = ar3
    return matrix3

def matrix_add(matrix1, matrix2):
    array1 = matrix1.matrixx
    array2 = matrix2.matrixx
    if type(matrix1) != matrix or type(matrix2) != matrix:
        raise TypeError("Inputs must be matrices")
    if matrix2.cols != matrix1.cols or matrix1.rows != matrix2.rows:
        raise ValueError("Dimensions not combatible")

    matrix3 = matrix(matrix1.rows, matrix1.cols)
    for j in range(matrix1.rows):
        for i in range(matrix1.cols):
            array3 = []
            array3.append(array1[j][i] + array2[j][i])
        matrix.matrixx.append(array3)
    return matrix3
def matrix_add_complex(matrix1,matrix2):
    array1 = matrix1.matrixx
    array2 = matrix2.matrixx
    if type(matrix1) != matrix or type(matrix2) != matrix:
        raise TypeError("Inputs must be matrices")
    if matrix2.cols != matrix1.cols or matrix1.rows != matrix2.rows:
        raise ValueError("Dimensions not combatible")
    matrix3 = matrix(matrix1.rows, matrix1.cols)
    for j in range(matrix1.rows):
        for i in range(matrix1.cols):
            array3 = []
            array3.append(add_complex(array1[j][i], array2[j][i]))
        matrix.matrixx.append(array3)
    return matrix3


def determinant_3d(a_matrix):
    vec1 = vector(a_matrix.matrixx[1])
    vec2 = vector(a_matrix.matrixx[2])
    vec3 = a_matrix.matrixx[0]
    determinant = [(vec1.array[1] * vec2.array[2] - vec1.array[2] * vec2.array[1]) * vec3[0],
                   (-vec1.array[0] * vec2.array[2] + vec1.array[2] * vec2.array[0]) * vec3[1],
                   (vec1.array[0] * vec2.array[1] - vec1.array[1] * vec2.array[0]) * vec3[2]
                   ]
    sum = 0
    for i in range(len(determinant)): sum += determinant[i]
    return sum


def determinant(a_matrix, d=0, show_current_det=False):
    if type(a_matrix) != matrix or a_matrix.rows != a_matrix.cols:
        raise TypeError("Input must be square nxn matrix")
    if a_matrix.rows == 1:
        return 0
    if a_matrix.rows == 2:
        return a_matrix.matrixx[0][0] * a_matrix.matrixx[1][1] - a_matrix.matrixx[0][1] * a_matrix.matrixx[1][0]
    else:
        sub_array = []
        new_matrix = matrix(a_matrix.rows - 1, a_matrix.cols - 1)
        # print("LINE301:",sub_array)
        for i in range(a_matrix.cols):
            new_matrix = matrix(a_matrix.rows - 1, a_matrix.cols - 1)
            new_matrix.matrixx = []
            for j in range(1, a_matrix.rows):

                # print("j is:",j)

                for k in range(i):
                    sub_array.append(a_matrix.matrixx[j][k])

                # print(sub_array)
                for k in range(i + 1, a_matrix.cols):
                    sub_array.append(a_matrix.matrixx[j][k])
                #   print(sub_array)
                new_matrix.matrixx.append(sub_array)
                sub_array = []
            # print(sub_array)

            # new_matrix.printf()
            if i % 2 == 0:
                sign = 1
            else:
                sign = -1
            # print("error:",d)
            d += sign * a_matrix.matrixx[0][i] * determinant(new_matrix)
            if show_current_det == True: print("current det is:", d)

        return d


def rref(A_matrix):
    # check for input compatability
    if type(A_matrix) != matrix or A_matrix.rows != A_matrix.cols:
        raise TypeError("Input must be n x n matrix")

    a_matrix = A_matrix.matrixx  # obtain array like object from matrix type object

    n = A_matrix.rows
    # gaussian elimination
    for k in range(n - 1):
        # row swapping if necessary
        # actual elimination
        for i in range(k + 1, n):
            if a_matrix[i][k] == 0: continue
            factor = a_matrix[k][k] / a_matrix[i][k]
            for j in range(k, n):
                a_matrix[i][j] = a_matrix[k][j] - a_matrix[i][j] * factor
    return (a_matrix)


def nabs(num):
    if num < 0:
        return -num
    else:
        return num

def raise_matrix(a_matrix, power):
    if type(a_matrix) != matrix or a_matrix.cols != a_matrix.rows:
        raise TypeError("Input must be a square nxn matrix")
    raised_matrix = a_matrix
    for i in range(power):
        raised_matrix = matrix_mult(raised_matrix, a_matrix)
    return raised_matrix


def matrix_by_scalar(a_matrix, scalar):
    if type(a_matrix) != matrix:
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
    # this function is optimized for commonly found 3x3 matrices and can only be used for these dimensions
    if type(a_matrix) != matrix or a_matrix.rows() != 3 or a_matrix.cols() != 3:
        raise TypeError("Input must be a 3x3 matrix")
    lambda_matrix = matrix(3, 3)
    determinant = determinant_3d(a_matrix)


def check_for_symmetry(a_matrix, stop=False):
    if type(a_matrix) != matrix or a_matrix.rows != a_matrix.cols:
        raise TypeError("Input must be a square nxn matrix")
    array = a_matrix.matrixx
    for r in range(a_matrix.rows):
        for c in range(a_matrix.cols):
            if r == c: continue
            if array[r][c] == array[c][r]: continue
            if stop == True: raise ValueError("Matrix not symmetric")
            return False
    return True


def normalise_matrix(a_matrix):
    if type(a_matrix) != matrix:
        raise TypeError("Input must be a square nxn matrix")
    det = determinant(a_matrix)
    if det == 0: raise ValueError("Matrix cannot be normalised")
    normalised_matrix = matrix_by_scalar(a_matrix, 1 / det)
    return normalised_matrix



def normalise_vector(a_vector):
    if type(a_vector) != vector:
        raise TypeError("Input must be a vector")
    sum = 0
    for i in range(a_vector.Length):
        sum += (a_vector.array[i]) ** 2
    norm = sum ** (0.5)
    for i in range(a_vector.Length):
        a_vector.array[i] *= 1 / norm
    return a_vector


def raise_matrix_comps(a_matrix, power):
    if type(a_matrix) != matrix:
        raise TypeError("input must be a matrix")
    for i in range(a_matrix.rows):
        for j in range(a_matrix.cols):
            a_matrix.matrixx[i][j] = a_matrix.matrixx[i][j] ** power
    return a_matrix




def cholesky(a_matrix):
    import numpy as np
    a = np.array(a_matrix.matrixx, float)
    L = np.zeros_like(a)
    n, _ = np.shape(a)
    for j in range(n):
        for i in range(j, n):
            if i == j:
                L[i, j] = np.sqrt(a[i, j] - np.sum(L[i, :j] ** 2))
            else:
                L[i, j] = (a[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    result = matrix(a_matrix.rows, a_matrix.cols)
    result.matrixx = L.tolist()
    return result



def power_iter(a_matrix, steps=10, value = False):
    b = matrix(a_matrix.cols, 1)
    b.matrixx = [[1] for i in range(a_matrix.cols)]
    eigen_value = 0
    for i in range(steps):
        prod = matrix_mult(a_matrix, b)
        eigen_value = prod.matrixx[0][0]
        norm = matrix_by_scalar(prod,1/prod.matrixx[0][0])
        b = norm
    if value == True :
        return eigen_value
    return b
def create_identity_matrix(dimension):
    Ar = []
    sub = []
    for i in range(dimension):
        for j in range(dimension):
            if i == j :
                sub.append(1)
            else:
                sub.append(0)
        Ar.append(sub)
        sub = []
    identity_matrix = matrix(dimension,dimension)
    identity_matrix.matrixx = Ar
    return identity_matrix
def copy_matrix(a_matrix):
    if type(a_matrix) != matrix :
        raise TypeError("Input must be a matrix object")
    copy  = matrix(a_matrix.rows , a_matrix.cols)
    copy.matrixx = []
    sub = []
    for i in range(a_matrix.rows):
        for j in range(a_matrix.cols):
            sub.append(a_matrix.matrixx[i][j])
        copy.matrixx.append(sub)
        sub = []
    return copy
def invert_matrix(a__matrix):
    a_matrix = copy_matrix(a__matrix)
    if determinant(a_matrix) == 0 :  #check for singularity
        raise ValueError("Matrix is singular , cannot be inverted")
    if type(a_matrix) != matrix or a_matrix.rows != a_matrix.cols :
        raise TypeError("Input must be a square nxn matrix")
    #initiate inverted matrix and corresponding identity matrix , in the end identity matrix will be the inversion of the input and the inverted_Matrix will be a identity matrix
    inverted = matrix(a_matrix.rows , a_matrix.rows)
    inverted.matrixx = a_matrix.matrixx
    identity_matrix = create_identity_matrix(a_matrix.rows)
    #scaling each row by the inverse of its diagonal element
    for fd in range(a_matrix.rows):
        diag_scaler = 1.0 / a_matrix.matrixx[fd][fd]
        for j in range(a_matrix.rows):
            inverted.matrixx[fd][j] *= diag_scaler
            identity_matrix.matrixx[fd][j] *= diag_scaler
        # reducing every row by the product of the element in the same col as the diagonal scalar  multiplied by the row of the current diagonal scalar
        for i in range(0 , fd):
            current_scaler = inverted.matrixx[i][fd]
            for k in range(a_matrix.rows):
                inverted.matrixx[i][k] -= current_scaler*inverted.matrixx[fd][k]
                identity_matrix.matrixx[i][k] -= current_scaler*identity_matrix.matrixx[fd][k]
        for i in range(fd+1 , a_matrix.rows ):
            current_scaler = inverted.matrixx[i][fd]
            for k in range(a_matrix.rows):
                inverted.matrixx[i][k] -= current_scaler*inverted.matrixx[fd][k]
                identity_matrix.matrixx[i][k] -= current_scaler*identity_matrix.matrixx[fd][k]
    return identity_matrix


def check_for_indeterminate(a_matrix,b_matrix):
    for i in range(a_matrix.rows):
        for j in range(a_matrix.cols):
            a_matrix.matrixx[i][j] *= 1/b_matrix.matrixx[i][0]
    for i in range(a_matrix.rows):
        for j in range(0,i):
            if a_matrix.matrixx[i] == a_matrix.matrixx[j] :
                raise ArithmeticError("Indeterminate system")
        for j in range(i+1,a_matrix.rows):
            if a_matrix.matrixx[i] == a_matrix.matrixx[j]:
                raise ArithmeticError("Indeterminate system")
    for i in range(a_matrix.rows):
         for j in range(a_matrix.cols):
            a_matrix.matrixx[i][j] *=  b_matrix.matrixx[i][0]
    return 1
def adjucate(a_matrix):
    copy = copy_matrix(a_matrix)
    return matrix_by_scalar(copy , determinant(copy))
def check_for_inconsistency(a_matrix,b_matrix):
    for i in range(a_matrix.rows):
        for j in range(0,i):
            if nabs(a_matrix.matrixx[j][0]/a_matrix.matrixx[i][0] - b_matrix.matrixx[j][0] / b_matrix.matrixx[i][0] ) > .00001:
                raise ArithmeticError("Inconsistent system")
        for j in range(i+1 , a_matrix.rows):
            if nabs(a_matrix.matrixx[j][0] / a_matrix.matrixx[i][0] - b_matrix.matrixx[j][0] / b_matrix.matrixx[i][0]) > .00001:
                raise ArithmeticError("Inconsistent system")
    return 1
def solve_system_using_inverse(a_matrix , b , indeterminate_check = False ):
    """
    WARNING: should only be used for a system with non-singular a_matrix
    :param a_matrix:
    :param b:
    :param indeterminate_check:
    :return: the solutions of the system in matrix form
    """
    ar = []
    b_matrix = matrix(a_matrix.rows,1)
    for i in range(len(b)):
        ar.append(b[i])
        b_matrix.matrixx.append(ar)
        ar = []
    if indeterminate_check == True:
          #  check_for_inconsistency(a_matrix,b_matrix) patch it up here
            check_for_indeterminate(a_matrix,b_matrix)
    if type(a_matrix) != matrix or a_matrix.rows != a_matrix.cols or len(b) != a_matrix.rows:
        raise TypeError("Input must be given as a nxn matrix , which represents the coefficients of the linear system")
    if determinant(a_matrix) == 0 :
        raise ArithmeticError("Matrix is singular , not invertible")
    x_matrix = matrix_mult(invert_matrix(a_matrix),b_matrix)
    return x_matrix
def solve_system(a_matrix: object, b_matrix: object) -> object:
    """
     Finds the solutions of a linear system. WARNING : b_matrix.shape -> nx1 !!!
    :param a_matrix:
    :param b_matrix:
    :return: the solutons of the system in matrix form
    """
    if type(a_matrix) != matrix or a_matrix.rows != a_matrix.cols or b_matrix.rows != a_matrix.rows:
        raise TypeError("Input must be given as a nxn matrix and a 1xn, of which the first represents the coefficients of the linear system , while the second the constant terms of the system")
    #check_for_indeterminate(a_matrix,b_matrix)
    AM = copy_matrix(a_matrix)
    BM = copy_matrix(b_matrix)
    n = AM.rows
    for fd in range(n):
        scaler = 1/AM.matrixx[fd][fd]
        for j in range(n):
            AM.matrixx[fd][j] *= scaler #scale each row by the inverse of each diagonal element
        BM.matrixx[fd][0] *= scaler
        for i in range(0,fd):
            crScaler = AM.matrixx[i][fd]  # cr stands for "current row"/ # reducing every row by the product of the element in the same col as the diagonal scalar  multiplied by the row of the current diagonal scalar
            for k in range(n):  # cr - crScaler*fdRow.
                AM.matrixx[i][k] = AM.matrixx[i][k] - crScaler * AM.matrixx[fd][k]
            BM.matrixx[i][0] = BM.matrixx[i][0] - crScaler * BM.matrixx[fd][0]
        for i in range(fd+1 , n):
            crScaler = AM.matrixx[i][fd]  # same as before , now for rows after the row of the current diagonal element
            for k in range(n):  # cr - crScaler*fdRow.
                AM.matrixx[i][k] = AM.matrixx[i][k] - crScaler * AM.matrixx[fd][k]
            BM.matrixx[i][0] = BM.matrixx[i][0] - crScaler * BM.matrixx[fd][0]
    #by completing the iterations , the AM matrix , namely the morphed copy of the coefficient matrix
    # is now an identity matrix, while the morphed copy of the b matrix is now the solutions matrix
    return BM

def round_elements(a_matrix,tolerance = 3 ):
    for i in range(a_matrix.rows):
        for j in range(a_matrix.cols):
            sign = 1
            if a_matrix.matrixx[i][j] < 0:
                sign = - 1
            if nabs(a_matrix.matrixx[i][j]) < .000001:
                a_matrix.matrixx[i][j] = 0
            if ((nabs(a_matrix.matrixx[i][j]) % 1) * (10 ** (tolerance))) % 1 > .5:

                a_matrix.matrixx[i][j] = sign * (nabs(a_matrix.matrixx[i][j]) // 1 + (
                            (((nabs((a_matrix.matrixx[i][j]) % 1) * (10 ** tolerance)) // 1) + 1) / (10 ** tolerance)))
            else:
                a_matrix.matrixx[i][j] = nabs(a_matrix.matrixx[i][j])*10 ** tolerance
                a_matrix.matrixx[i][j] = nabs(a_matrix.matrixx[i][j]) // 1
                a_matrix.matrixx[i][j] = nabs(a_matrix.matrixx[i][j])/ 10 ** tolerance
                a_matrix.matrixx[i][j] *=sign
    return a_matrix


def round_complex_elements(c_num,tolerance = 3 ):

    c_num.real = round(c_num.real,tolerance) ; c_num.im = round(c_num.im,tolerance)
    return c_num

def round_complex_matrix_elements(c_matrix , tolerance = 3):
    for i in range(c_matrix.rows):
        for j in range(c_matrix.cols):
            round_complex_elements(c_matrix.matrixx[i][j])
    return c_matrix



import math
def dft(f_array):
    if type(f_array) != list :
        raise TypeError("Input must be a nx1 array")
    f_matrix = matrix( len(f_array) , 1)
    for i in range(len(f_array)):
        f_matrix.matrixx.append([complex_num(vector([f_array[i],0]))])
    n = len(f_array)
    w = -2*math.pi/n
    freq_matrix = matrix(n,n)
    sub = []
    freq_matrix.matrixx = []
    for k in range(n):
        for j in range(n):
            #print("k,j is : ",k,j,"\n")
            sub.append(round_complex_elements(exp_complex(w*k*j)))
        freq_matrix.matrixx.append(sub)
        sub = []
    dft_matrix = matrix_mult_complex(freq_matrix,f_matrix)
    return dft_matrix
import numpy as np



def _main():
    if __name__ == "__main__":
        print("cuurently in module")
    else:
        print("Using linear algebra module")
