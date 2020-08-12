class rk4:
    def __init__(self,function,time_interval,initial_condition,dt=None ):
        self.function = function
        self.time_interval = time_interval
        if dt == None :
            self.dt =.001
        else:
            self.dt = dt
        self.initial_condition = initial_condition
        self.path_array = []
        self.trans_val = False

    def __rk4_single_step(self,tk,yk):
        f1 = self.function(tk, yk)
        f2 = self.function(tk + self.dt / 2, [yk[i] + (self.dt / 2) * f1[i] for i in range(len(yk))])
        f3 = self.function(tk + self.dt / 2, [yk[i] + (self.dt / 2) * f2[i] for i in range(len(yk))])
        f4 = self.function(tk + self.dt, [yk[i] + (self.dt / 2) * f3[i] for i in range(len(yk))])
        return [yk[i] + (self.dt / 6) * (f1[i] + 2 * (f2[i] + f3[i]) + f4[i]) for i in range(len(yk))]

    def __rk4_solve(self):
        yk = self.initial_condition
        for i in range(len(self.time_interval)):
            tk = self.time_interval[i]
            y_k_1 = self.__rk4_single_step( tk, yk)
            self.path_array.append(yk)
            yk = y_k_1
        return self.path_array

    def __tranpose_path(self):
        from linear_algebra_2 import matrix, transpose
        result_matrix = matrix(len(self.path_array),len(self.path_array[0]))
        result_matrix.matrixx = self.path_array
        self.path_array = transpose(result_matrix).matrixx
        return self.path_array

    def rk4_solve(self):
        self.__rk4_solve()
        self.__tranpose_path()
        return self.path_array


class rk78:
    def __init__(self,function,time_interval,initial_condition,dt=None):
        """
        :param function: function to be integrated
        :param time_interval: time interval in which the system is examined
        :param initial_condition: initial state of the system
        :param dt: size of discrete time steps
        """
        self.function = function
        self.time_interval = time_interval
        self.initial_condition = initial_condition
        self.path_array = []
        if dt == None :
            self.dt = .001
        else:
            self.dt = dt

    def  __ode78_single_step(self, yk, tk):
         f1 = [self.dt * self.function(tk,yk)[j] for j in range(len(yk))]
         f2 = [self.dt * self.function(tk+.5*self.dt , [yk[i] + .5*f1[i] for i in range(len(yk))])[j] for j in range(len(yk))]
         f3 = [self.dt * self.function(tk+.5*self.dt , [yk[i] + 1/4 * f1[i] + 1/4 * f2[i] for i in range(len(yk))])[j] for j in range(len(yk))]
         f4 = [self.dt * self.function(tk+self.dt,[yk[i] -f2[i] + 2*f3[i] for i in range(len(yk))])[j] for j in range(len(yk))]
         f5 = [self.dt * self.function(tk+ (2/3)*self.dt , [yk[i] + 7/27 * f1[i] + 10/27 * f2[i] + 1/27 * f4[i] for i in range(len(yk))])[j] for j in range(len(yk))]
         f6 = [self.dt * self.function( tk + 1/5 * self.dt  , [yk[i] + 28/625 * f1[i] -1/5 * f2[i] + 546/625 * f3[i] + 54/625 * f4[i] -378/625 * f5[i] for i in range(len(yk))])[j] for j in range(len(yk))]
         return [yk[i] + 1/24 * f1[i] + 5/48 * f4[i] + 27/56 *f5[i] + 125/336 * f6[i] for i in range(len(yk))]

    def __ode78(self):
         yk = self.initial_condition
         for i in range(len(self.time_interval)):
                   tk = self.time_interval[i]
                   yk_1 = self.__ode78_single_step(yk,tk)
                   self.path_array.append(yk)
                   yk = yk_1
         return self.path_array

    def __transpose(self):
        from linear_algebra_2 import matrix , transpose
        path_matrix = matrix(len(self.path_array),len(self.path_array[0]))
        path_matrix.matrixx = self.path_array
        self.path_array = transpose(path_matrix).matrixx
        return self.path_array
    def rk78_solve(self):
        """
        :return: the value of each state of the system in the time interval defined above
        """
        self.__ode78()
        self.__transpose()
        return self.path_array
