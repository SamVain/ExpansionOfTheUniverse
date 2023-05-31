import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants for a more realistic Universe
H_0 = 70  # Hubble constant at present time, in km/s/Mpc
Omega_m = 0.3  # Fractional matter density
Omega_r = 1e-5  # Fractional radiation density
Omega_lambda = 0.7  # Fractional dark energy density
Omega_k = 1 - Omega_m - Omega_r - Omega_lambda  # Fractional curvature density
G = 6.67430e-11  # Gravitational constant, in m^3 kg^-1 s^-2
# Critical density at present time, in kg/m^3
rho_crit_0 = 3 * H_0**2 / (8 * np.pi * G)
rho_m_0 = Omega_m * rho_crit_0  # Matter density at present time, in kg/m^3
rho_r_0 = Omega_r * rho_crit_0  # Radiation density at present time, in kg/m^3

# Friedmann equation: (da/dt)^2 = 8*pi*G/3 * (rho_m_0*(a_0/a)^3 + rho_r_0*(a_0/a)^4) + Omega_lambda*H_0**2*a_0**2 - Omega_k*H_0**2*a_0**2*(a_0/a)^2
# We will solve this equation numerically to find a(t)

# First, we create a function for the derivative in the equation


def da_dt(t, a):
    return np.sqrt(8 * np.pi * G / 3 * (rho_m_0 / a**3 + rho_r_0 / a**4) + Omega_lambda * H_0**2 * a**2 - Omega_k * H_0**2 * a)


# Time span
t_span = [0, 1e17]  # Time since Big Bang, in seconds

# Initial conditions
a_0 = 1
initial_conditions = [a_0]

# Solve the differential equation using the Runge-Kutta method
solution = solve_ivp(da_dt, t_span, initial_conditions, method='RK45')

# Plot the expansion of the Universe
plt.plot(solution.t, solution.y[0])
plt.xlabel('Time since Big Bang (s)')
plt.ylabel('Scale factor a(t)')
plt.title('Expansion of the Universe')
plt.show()
