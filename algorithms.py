# IOE 511/MATH 562, University of Michigan
# Code written by: Albert S. Berahas & Jiahao Shi

import numpy as np

# Function setting the step size
def set_step_size(x, f, g, d, problem, method, options):
    if method.step_type == 'Constant':
        return method.constant_step_size
    
    elif method.step_type == 'Backtracking':
        # Set initial step size
        alpha = method.constant_step_size
        while problem.compute_f(x+alpha*d) > f + method.c_1 * alpha * (g.T @ d).item():
            alpha *= method.tau
        return alpha
    else:
        raise ValueError(f"step type '{method.step_type}' not specified")
    
# Function for Cholesky algorithm
def CholeskyAlgorithm(H, beta):
    # Initialize eta
    diag_min = np.min(np.diag(H))
    if diag_min > 0:
        eta = 0
    else:
        eta = -diag_min + beta

    # Initialize flag
    L = None

    while L is None:
        try:
            L = np.linalg.cholesky(H + eta * np.eye(len(H)))
        except np.linalg.LinAlgError:
            eta = max(2*eta, beta)
    return H, eta

def finite_diff_grad(x, g, problem, method, options):
    g_approx = np.zeros((problem.n, 1))
    for i in range(problem.n):
        e = np.zeros((problem.n, 1))
        e[i] = options.epsilon

        x_ub = x + e
        x_lb = x - e

        g_approx[i] = (problem.compute_f(x_ub) - problem.compute_f(x_lb)) / (2 * options.epsilon)
    print(np.linalg.norm(g_approx - g))
    if np.linalg.norm(g_approx - g) < options.epsilon:
        return True
    else:    
        return False
    
    
def finite_diff_Hess(x, H, problem, method, options):
    H_approx = np.zeros((problem.n, problem.n))
    for i in range(problem.n):
        for j in range(problem.n):
            ei = np.zeros((problem.n, 1))
            ej = np.zeros((problem.n, 1))

            ei[i] = options.epsilon
            ej[j] = options.epsilon

            H_approx[i, j] = (problem.compute_f(x + ei + ej) - problem.compute_f(x + ei) - problem.compute_f(x + ej) + problem.compute_f(x)) / (options.epsilon**2) 
    if np.linalg.norm(H_approx - H, ord='fro') < options.epsilon:
        return True
    else:
        return False

# Compute the next step for all iterative optimization algorithms given current solution x:
# (1) Gradient Descent
# (2) Newton's method
# (3) BFGS
# (4) L-BFGS


def GDStep(x, f, g, problem, method, options):
    # Reshape x to be column vector
    x = x.reshape(-1, 1)

    # Check if the gradient is calculated correctly
    if options.check_grad:
        flag = finite_diff_grad(x, g, problem, method, options)
        if not flag:
            print("The gradient is not calculated correctly")

    # Set the search direction d to be -g
    d = -g
    
    # determine step size
    alpha = set_step_size(x, f, g, d, problem, method, options)
    x_new = x + alpha*d 
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    return x_new, f_new, g_new, d, alpha

def NewtonStep(x, f, g, H, problem, method, options):
    # Reshape x to be column vector
    x = x.reshape(-1, 1)

    # Check if the gradient is calculated correctly
    if options.check_grad:
        grad_flag = finite_diff_grad(x, g, problem, method, options)
        if not grad_flag:
            print("The gradient is not calculated correctly")

    # Check if the Hessian is calculated correctly
    if options.check_Hess:
        Hess_flag = finite_diff_Hess(x, H, problem, method, options)
        if not Hess_flag:
            print("The Hessian is not calculated correctly")

    # Set the search direction d to be -H\g
    H, eta = CholeskyAlgorithm(H, method.beta)
    H = H + eta * np.eye(len(H))
    d = -np.linalg.solve(H, g)
    
    # determine step size
    alpha = set_step_size(x, f, g, d, problem, method, options)
    x_new = x + alpha*d 
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)
    H_new = problem.compute_H(x_new)

    return x_new, f_new, g_new, H_new, d, alpha

def BFGSStep(x, f, g, quasi_H, problem, method, options):
    # Reshape x to be column vector
    x = x.reshape(-1, 1)

    # Check if the gradient is calculated correctly
    if options.check_grad:
        flag = finite_diff_grad(x, g, problem, method, options)
        if not flag:
            print("The gradient is not calculated correctly")
    
    # Set the search direction d to be -H * g
    d = - quasi_H @ g
    
    # determine step size
    alpha = set_step_size(x, f, g, d, problem, method, options)
    x_new = x + alpha*d 
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    return x_new, f_new, g_new, d, alpha

def L_BFGSStep(x, f, g, m, H_0, s_k_hist, y_k_hist, problem, method, options):
    # Reshape x to be column vector
    x = x.reshape(-1, 1)

    # Check if the gradient is calculated correctly
    if options.check_grad:
        flag = finite_diff_grad(x, g, problem, method, options)
        if not flag:
            print("The gradient is not calculated correctly")

    # Initiate alpha memory
    alpha_hist = []

    # Set q = nabla f(x)
    q = g.copy()

    # For the first loop, iterate s_k and y_k backward
    for s_i, y_i in zip(reversed(s_k_hist), reversed(y_k_hist)):
        rho_i = 1 / (s_i.T @ y_i).item()
        alpha_i = rho_i * (s_i.T @ q).item()
        q -= alpha_i * y_i

        alpha_hist.append(alpha_i)

    alpha_hist.reverse()

    # At the end of first loop, set r to be H_0 @ q
    r = H_0 @ q 

    # Starting second loop
    for alpha_i, s_i, y_i in zip(alpha_hist, s_k_hist, y_k_hist):
        rho_i = 1 / (s_i.T @ y_i).item()
        beta = rho_i * (y_i.T @ r).item()
        r += s_i * (alpha_i - beta)

    # Set the search direction d to be - r since r = H_k @ nabla f(x)
    d = - r 

    # determine step size
    alpha = set_step_size(x, f, g, d, problem, method, options)
    x_new = x + alpha*d 
    f_new = problem.compute_f(x_new)
    g_new = problem.compute_g(x_new)

    return x_new, f_new, g_new, d, alpha