import numpy as np


def laplace(Field, c1, c2, c3, c4, dx, dy, tolerance=1e-4):
    """
    Numerical routine for the Laplace equation : dU^2/dx^2 + dU^2/dy^2 = 0
    Uses Finite Differnce Method
    :param Field: 2d array which holds the value of each grid node
    :param c1: boundary condition 1
    :param c2: b.c.2
    :param c3: b.c.3
    :param c4: b.c.4
    :param dx: distance between nodes on x-axs
    :param dy: distance between nodes on y-axis
    :param tolerance: error tolerance-loop exits when the error is smaller than this variable
    :return: The update Field (with updated values for each node)
    """
    error = 1
    while (error > tolerance):
        field0 = Field.copy()
        Field[1:-1, 1:-1] = ((dy ** 2) * (field0[1:-1, 2:] + field0[1:-1, 0:-2]) + (dx ** 2) * (
                    field0[2:, 1:-1] + field0[:-2, 1:-1])) \
                            / (2 * (dx ** 2 + dy ** 2))
        Field[:, 0] = c1
        Field[:, -1] = c2
        Field[0, :] = c3
        Field[-1, :] = c4
        error = (np.sum(np.abs(Field) - np.abs(field0))) / np.sum(np.abs(field0))
    return Field


def __heat_2d_timestep(Field, field0, dx, dy, dt, D, c1, c2, c3, c4):
    Field[1:-1, 1:-1] = field0[1:-1, 1:-1] + D * dt * (
                (field0[2:, 1:-1] - 2 * field0[1:-1, 1:-1] + field0[:-2, 1:-1]) / dx ** 2 + (
                    field0[1:-1, 2:] - 2 * field0[1:-1, 1:-1] + field0[1:-1, :-2]) / dy ** 2)
    Field[:, 0] = c1
    Field[:, -1] = c2
    Field[0, :] = c3
    Field[-1, :] = c4
    return Field

def __heat_1d_timestep(Field,field0,dx,dt,sigma,nu,c1,c2):
    Field[1:-1] = field0[1:-1] + nu * dt * (
            (field0[2:] - 2 * field0[1:-1] + field0[:-2]) / dx ** 2 )
    Field[0] = c1
    Field[-1] = c2
    return Field

def heat_1d(Field , sigma , nu  , tsteps , nsnaps , dx , c1 , c2):
    """
    :param Field: 1d vector of values of each grid node
    :param sigma: diffusion coefficient 1
    :param nu   : diffusion coefficient 2
    :param tsteps: Number of time steps or iteration to be performed
    :param nsnaps: Number of snapshots to keep
    :param dx: distance between nodes on x-axis
    :param c1: boundary condition 1
    :param c2: b.c.2
    :return: 2d of snapshots taken. Each element of this array is a vector which holds the value of each node at said time
    """
    dt = (sigma/nu) * (dx**2)
    time_matrix = []
    for i in range(tsteps):
        field0 = Field.copy()
        if i in nsnaps:
            time_matrix.append(field0)
        __heat_1d_timestep(Field,field0,dx,dt,sigma,nu,c1,c2)
    return time_matrix

def heat_2d(Field, D, tsteps, nsnaps, dx, dy, c1, c2, c3, c4):

    """
    Numerical Method for the 2d heat or diffusion equation (no heat transfer):
    D*dT/dt = dT^2/dx^2 + dT^2/dy^2
    :param Field: 2d array which holds the value of each grid node
    :param D: Coefficient of diffusion
    :param tsteps: Number of time steps , or iterations to be performed
    :param nsnaps: Number of snapshots to keep
    :param c1: boundary condition 1
    :param c2: b.c.2
    :param c3: b.c.3
    :param c4: b.c.4
    :param dx: distance between nodes on x-axs
    :param dy: distance between nodes on y-axis
    :return: 3d array which holds nsnaps number of 2d grid matrices.Each of these sub-matrices contains the values of each grid node at said time-step
    """

    dt = (1 / (2 * D)) * (dx * dy) ** 2 / ((dx ** 2) + (dy ** 2))
    time_matrix = []
    for i in range(tsteps):
        field0 = Field.copy()
        if i in nsnaps:
            time_matrix.append(field0)
        __heat_2d_timestep(Field, field0, dx, dy, dt, D, c1, c2, c3, c4)
    return time_matrix
