# IOE 511/MATH 562, University of Michigan
# Framework code written by: Albert S. Berahas & Jiahao Shi

# Function that runs a chosen algorithm on a chosen problem
#           Inputs: problem, method, options (structs)
#           Outputs: final iterate (x) and final function value (f)
import numpy as np
import functions
from algorithms import GDStep, NewtonStep, BFGSStep, L_BFGSStep

def BFGS_update(H_k, y_k, s_k):
    I = np.eye(len(H_k))
    denom = 1/(s_k.T @ y_k)[0, 0]
    H_kp1 = (I - denom * (s_k @ y_k.T)) @ H_k @ (I - denom * (y_k @ s_k.T)) + (denom * (s_k @ s_k.T))
    return H_kp1

def optSolver(problem,method,options):
    # compute initial function/gradient/Hessian
    x = problem.x0 if problem.x0.ndim == 2 else problem.x0.reshape(-1, 1)
    f = problem.compute_f(x)
    g = problem.compute_g(x)
    H = problem.compute_H(x)
    norm_g = np.linalg.norm(g,ord=np.inf)

    # Compute conditional for termination
    cond = max(norm_g, 1)

    # set initial iteration counter
    k = 0
    
    # Initiate s_k and y_k memory in case of L-BFGS
    s_k_hist = []
    y_k_hist = []

    # Initiate empty array to store f and g
    f_hist = []
    g_hist = []

    while norm_g > options.term_tol * cond and k < options.max_iterations:
        if method.name == 'GradientDescent':
            x_new, f_new, g_new, d, alpha = GDStep(x,f,g,problem,method,options)

        elif method.name == 'Newton': 
            x_new, f_new, g_new, H_new, d, alpha = NewtonStep(x,f,g,H,problem,method,options)
            H = H_new

        elif method.name == 'BFGS':
            # set quasi_H = I for the first iteration
            if k == 0:
                quasi_H = np.eye(problem.n)
            else:
                s_k = x - x_old
                y_k = g - g_old

                # Update if denominator is sufficiently positive
                if (s_k.T @ y_k).item() >= options.term_tol * np.linalg.norm(s_k.flatten(), ord=2) * np.linalg.norm(y_k.flatten(), ord=2):
                    quasi_H = BFGS_update(quasi_H, y_k, s_k)

            x_new, f_new, g_new, d, alpha = BFGSStep(x,f,g,quasi_H,problem,method,options)

        elif method.name == 'L-BFGS':
            m = method.m
            H_0 = np.eye(problem.n)
            if k > 0:
                s_k = x - x_old
                y_k = g - g_old
                if (s_k.T @ y_k).item() >= options.term_tol * np.linalg.norm(s_k.flatten(), ord=2) * np.linalg.norm(y_k.flatten(), ord=2):
                    s_k_hist.append(x - x_old)
                    y_k_hist.append(g - g_old)
            
            if len(s_k_hist) > m or len(y_k_hist) > m:
                s_k_hist.pop(0)
                y_k_hist.pop(0)

            x_new, f_new, g_new, d, alpha = L_BFGSStep(x, f, g, m, H_0, s_k_hist, y_k_hist, problem, method, options)

        else:
            print('Warning: method is not implemented yet')
            
        if options.verbose:
            with np.printoptions(precision=5, suppress=True):
                print(f"At step {k} | f = {f}, grad = {norm_g}, alpha = {alpha}")

        # Store f and g
        f_hist.append(f)
        g_hist.append(norm_g)
            
        # update old and new function values        
        x_old = x; f_old = f; g_old = g; norm_g_old = norm_g
        x = x_new; f = f_new; g = g_new; norm_g = np.linalg.norm(g.flatten(),ord=np.inf)

        # increment iteration counter
        k = k + 1 

    return x, f, f_hist, g_hist