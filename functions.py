# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi

import numpy as np

# Define all the functions and calculate their gradients and Hessians, those functions include:
# (1) Rosenbrock function
# (2) Quadractic function
# (3) Function 2
# (4) Function 3


### Rosenbrock problem ###

# Function that computes the function value for the Rosenbrock function
#
#           Input: x
#           Output: f(x)
#

def rosen_func(x):
    if x.ndim > 1:
        x = x.flatten()
    f = (1-x[0])**2 + 100*(x[1] - x[0]**2)**2
    return f

# Function that computes the gradient of the Rosenbrock function
#
#           Input: x
#           Output: g = nabla f(x)
#

def rosen_grad(x):
    if x.ndim > 1:
        x = x.flatten()
    g = np.array([[2*(x[0]-1)-400*x[0]*(x[1]-x[0]**2)], [200*(x[1]-x[0]**2)]])
    return g


# Function that computes the Hessian of the Rosenbrock function
#
#           Input: x
#           Output: H = nabla^2 f(x)
#

def rosen_Hess(x):
    if x.ndim > 1:
        x = x.flatten()
    H = np.array([[2-400*x[1]+1200*x[0]**2, -400*x[0]],
                  [-400*x[0], 200]])
    return H 

### Function 2 ###
# Function that computes the function value for the function 2
#
#           Input: x
#           Output: f(x)
#

def f2_func(x):
    if x.ndim > 1:
        x = x.flatten()
    # Define y
    y = [1.5, 2.25, 2.625]

    f = 0
    for i in range(1, len(y)+1):
        f += (y[i-1] - x[0]*(1-(x[1]**i)))**2
    return f


# Function that computes the gradient value for the function 2
#
#           Input: x
#           Output: nabla f(x)
#

def f2_grad(x):
    if x.ndim > 1:
        x = x.flatten()

    # Define y
    y = [1.5, 2.25, 2.625]

    # Initialize g
    g = np.zeros((2, 1))
    
    # Increment g
    for i in range(1, len(y)+1):
        g += 2 * np.array([[(y[i-1]-x[0]*(1-x[1]**i)) * ((x[1]**i)-1)],
                           [(y[i-1]-x[0]*(1-x[1]**i)) * (x[0]*i*(x[1]**(i-1)))]])
    return g

# Function that computes the Hessian matrix for the function 2
#
#           Input: x
#           Output: nabla^2 f(x)
#

def f2_Hess(x):
    if x.ndim > 1:
        x = x.flatten()
        
    # Define y
    y = [1.5, 2.25, 2.625]

    # Initialize H
    H = np.zeros((2, 2))

    # Increment H
    for i in range(1, len(y)+1):
        H += 2 * np.array([[(1-x[1]**i)**2, y[i-1]*(i*(x[1]**(i-1))) + 2*x[0]*((x[1]**i)-1)*(i*(x[1]**(i-1)))],
                           [y[i-1]*(i*(x[1]**(i-1))) + 2*x[0]*((x[1]**i)-1)*(i*(x[1]**(i-1))), 0]])
    
    H[1, 1] = 2 * ((x[0]**2) + 2*(y[1]*x[0] + (3*(x[1]**2 - 1)*(x[0]**2))) + 3*(2*y[2]*x[1]*x[0] + (5*(x[1]**4) - 2*x[1])*(x[0]**2)))

    return H

### Function 3 ###
# Function that computes the function value for the function 3
#
#           Input: x
#           Output: f(x)
#

def f3_func(x):
    if x.ndim > 1:
        x = x.flatten()

    n = len(x)
    
    f = ((np.exp(x[0])-1) / (np.exp(x[0])+1)) + 0.1*np.exp(-x[0])
    for i in range(1, n):
        f += (x[i]-1)**4

    return f


# Function that computes the gradient value for the function 3
#
#           Input: x
#           Output: nabla f(x)
#

def f3_grad(x):
    if x.ndim > 1:
        x = x.flatten()

    n = len(x)

    g = np.zeros((n, 1))

    g[0, 0] = ((2*np.exp(x[0])) / ((np.exp(x[0])+1)**2)) - (0.1*np.exp(-x[0]))
    for i in range(1, n):
        g[i, 0] = 4*(x[i]-1)**3

    return g

# Function that computes the Hessian matrix for the function 3
#
#           Input: x
#           Output: nabla^2 f(x)
#

def f3_Hess(x):
    if x.ndim > 1:
        x = x.flatten()
    
    n = len(x)
    H = np.zeros((n, n))

    H[0, 0] = ((2*np.exp(x[0])*(1-np.exp(x[0]))) / ((np.exp(x[0])+1)**3)) + (0.1*np.exp(-x[0]))
    for i in range(1, n):
        H[i, i] = 12 * (x[i]-1)**2

    return H



# Function that computes the function value for the Quadractic function
#
#           Input: x
#           Output: f(x)
#

def quad_func(x):
    return ### Add code!  