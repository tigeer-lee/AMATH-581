# -*- coding: utf-8 -*-
"""
Modified version using np.trapz for integration and corrected boundary conditions
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define spatial range and step size
L = 4
dx = 0.1
x_positive = np.arange(0, L + dx, dx)  # x from 0 to L
x_full = np.arange(-L, L + dx, dx)     # x from -L to L
num_points = len(x_full)

# Initialize arrays to store eigenfunctions and eigenvalues
A1 = np.zeros((num_points, 5))
A2 = np.zeros(5)

# Define the differential equation
def schrodinger(x, y, epsilon):
    # y[0] = φ, y[1] = φ'
    dydx = [y[1], (x**2 - epsilon) * y[0]]
    return dydx

# Define the shooting method function
def shooting(epsilon, parity):
    # Set initial conditions based on parity
    if parity == 'even':
        y0 = [1.0, 0.0]  # φ(0) = 1, φ'(0) = 0
    else:
        y0 = [0.0, 1.0]  # φ(0) = 0, φ'(0) = 1

    sol = solve_ivp(schrodinger, [0, L], y0, args=(epsilon,), t_eval=[L], method='RK45')
    y_L = sol.y[:, -1]
    phi_L = y_L[0]
    phi_prime_L = y_L[1]
    # Use the boundary condition φ'(L) + L φ(L) = 0
    S = phi_prime_L + L * phi_L
    return S

# Start solving for the first five eigenvalues and eigenfunctions
for n in range(5):
    # Determine parity
    if n % 2 == 0:
        parity = 'even'  # Even n corresponds to even functions
    else:
        parity = 'odd'   # Odd n corresponds to odd functions

    # Initialize energy guess values
    epsilon_lower = 2 * n + 0.5  # Lower bound
    epsilon_upper = 2 * n + 1.5  # Upper bound

    # Calculate initial S values
    S_lower = shooting(epsilon_lower, parity)
    S_upper = shooting(epsilon_upper, parity)

    # Use the bisection method to find the appropriate ε_n
    tol = 1e-6
    max_iter = 100
    for _ in range(max_iter):
        epsilon = (epsilon_lower + epsilon_upper) / 2
        S = shooting(epsilon, parity)
        if abs(S) < tol:
            break  # Found appropriate ε_n
        elif S * S_lower < 0:
            epsilon_upper = epsilon
            S_upper = S
        else:
            epsilon_lower = epsilon
            S_lower = S

    # Store eigenvalue
    A2[n] = epsilon

    # Compute the complete eigenfunction
    if parity == 'even':
        y0 = [1.0, 0.0]
    else:
        y0 = [0.0, 1.0]

    sol = solve_ivp(schrodinger, [0, L], y0, args=(epsilon,), t_eval=x_positive, method='RK45')
    phi_positive = sol.y[0]

    # Extend to the negative half-axis based on parity
    if parity == 'even':
        phi_full = np.concatenate((phi_positive[::-1], phi_positive[1:]))
    else:
        phi_full = np.concatenate((-phi_positive[::-1], phi_positive[1:]))

    # Normalize using np.trapz
    norm = np.trapz(phi_full**2, x_full)
    phi_full_normalized = phi_full / np.sqrt(norm)

    # Store the absolute value of the eigenfunction
    A1[:, n] = np.abs(phi_full_normalized)

# Plot the absolute values of the eigenfunctions
for n in range(5):
    plt.plot(x_full, A1[:, n], label=f'n={n+1}')

print("A1 matrix:")
print(A1)

# Set plot parameters
plt.xlabel('x')
plt.ylabel('$|\phi_n(x)|$')
plt.title('Absolute values of the first five eigenfunctions')
plt.legend()
plt.grid(True)
plt.show()

# Output eigenvalues
print("Eigenvalues:")
print(A2)
