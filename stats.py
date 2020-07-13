from linear_algebra_2 import matrix , copy_matrix  , transpose  , solve_system , matrix_mult


def least_squares_fit_2d(x_matrix , y_matrix  ):
    """

    :param x_matrix: parameters
    :param y_matrix: target data
    :return: best fitting coefficients in matrix form
    """
    if type(x_matrix) == matrix :
        if x_matrix.rows != 1 :
            raise ValueError("Matrix must be 1d")
        x = x_matrix.matrixx[0]; N = len(x_matrix.matrixx[0])
    else:
        x = x_matrix ; N = len(x_matrix)
    if type(y_matrix) == matrix :
        if y_matrix.rows != 1 :
            raise ValueError("Matrix must be 1d")
        y = y_matrix.matrixx[0] ; n = y_matrix.cols
    else:
        y = y_matrix ; n = len(y_matrix)
    if n != N :
        raise ValueError("Both input arrays/matrices must have the same length ")
    T = 0 ; U = 0 ; V= 0 ; W = 0
    for i in range(N):
        T += x[i]**2
        U +=  x[i]
        V += y[i]*x[i]
        W += y[i]
    m = (V*N - W*U)/(T*N - U**2)
    b = (V*U - W*T)/(U**2 - N*T)
    return [m + 0  , b + 0 ]

def least_sqaures_fit(X_matrix,Y_matrix):
    """
    This is the general algorithm for the least squares method ,
    namely the method for minimizing the square of the difference of actual data vs preditcted data.
    It treats the task as a system solving problem. An extra column is added to the parameter matrix , representing the constant terms,
    and then the system (XTX)*W = (XTY) is solved.
    :param X_matrix: parameters
    :param Y_matrix: target data
    :return: best fitting coefficients in matrix form
    """
    if type(X_matrix) != matrix or type(Y_matrix) != matrix :
        raise ValueError("Both inputs must be matrices")
    if Y_matrix.rows < Y_matrix.cols:
         y_matrix = transpose(Y_matrix)
    else:
         y_matrix = Y_matrix
    if X_matrix.rows < X_matrix.cols:
         x_matrix = transpose(X_matrix)
    else:
         x_matrix = X_matrix
    for i in range(x_matrix.rows):
        x_matrix.matrixx[i].append(1)
    x_matrix.cols +=1
    x_matrix_transpose = transpose(x_matrix)
    A = matrix_mult(x_matrix_transpose, x_matrix)
    B = matrix_mult(x_matrix_transpose, y_matrix)
    W = solve_system(A,B)
    return W





