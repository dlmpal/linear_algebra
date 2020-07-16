from linear_algebra_2 import matrix , copy_matrix  , transpose  , solve_system , matrix_mult , vector , nabs , dot_product


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



class Perceptron:
    """
    This is a simple neural network , that works using a single layer perceptron.
    The perceptron is activated if the weighted sum of the dot product of the input matrix with the
    weights is greater than a threshold value.
    1. Each row of the input matrix is multiplied by the weights
    2. The error is calculated through the sum of the above product and the element of the last col of each row of the input matrix
    3.The weights are adjusted by adding onto them the product of the learning rate , the error and the corresponding matrix element.
    i.e : weight[j] = lr*er*matrix[i][j] , where i the current row , and j the current row of the matrix
    Each weight is affected by the matrix elements which belong to the col with same indexing.
    4.Repeat from step for all epochs

    """
    def __init__(self,input_matrix,intial_weights):
        self.input_matrix = input_matrix # input matrix should be of matrix class
        self.initial_weights = intial_weights # weights can either be a list or matrix object
        if type(self.initial_weights) == matrix :
            self.initial_weights = self.initial_weights.matrixx

    def predict(self,input,weights,threshold = 0.0 ):
        activation = dot_product(vector(input) , vector(weights))
        return 1 if activation >= threshold else 0

    def accuracy(self, input_matrix , weights):
        nums_correct = 0
        predictions = []
        for i in range(input_matrix.rows):
            pred = self.predict(input_matrix.matrixx[i][:-1],weights)
            predictions.append(pred)
            if nabs(pred - input_matrix.matrixx[i][-1])< .00000001:
                nums_correct +=1
        print(nums_correct)
        return nums_correct/input_matrix.rows
    def train(self,epochs=10,learning_rate = 1.0, out = 0 , break_early = 0):
        weights = self.initial_weights
        for epoch in range(epochs):
            current_accuracy = self.accuracy(self.input_matrix,weights)
            if break_early == True and current_accuracy == 1.0:
                return weights
            if out == True :
                print("EPOCH",epoch ," .Current accuracy is :",current_accuracy,"\n")
            for i in range(self.input_matrix.rows):
                prediction  = self.predict(self.input_matrix.matrixx[i][:-1],weights) #predicted classification
                error = self.input_matrix.matrixx[i][-1] - prediction  # difference of real value vs prediction
                for j in range(len(weights)):
                    weights[j] += learning_rate*error*self.input_matrix.matrixx[i][j] # calculation of new weights to optimize classification
            print("Current weight values are:\n")
            print(weights)
        return weights


