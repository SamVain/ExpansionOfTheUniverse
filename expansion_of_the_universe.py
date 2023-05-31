import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants for a more realistic Universe
Omega_m = 0.3  # Fractional matter density
Omega_r = 1e-5  # Fractional radiation density
Omega_lambda = 0.7  # Fractional dark energy density
Omega_k = 1 - Omega_m - Omega_r - Omega_lambda  # Fractional curvature density

# Friedmann equation: (da/dt)^2 = Omega_m/a + Omega_r/a^2 + Omega_lambda*a^2 - Omega_k
# We will solve this equation numerically to find a(t)

# First, we create a function for the derivative in the equation
def da_dt(t, a):
    return np.sqrt(Omega_m / a + Omega_r / a**2 + Omega_lambda * a**2 - Omega_k)

# Initial conditions
a_0_past_present = 1e-10  # The Universe was much smaller in the past
a_0_future = 1  # At present time, the Universe is of size 1

# Time span for the past and present
t_span_past_present = [0, 1]  # From Big Bang to present time

# Solve the differential equation for the past and present
solution_past_present = solve_ivp(da_dt, t_span_past_present, [a_0_past_present], method='RK45')

# Time span for the future
t_span_future = [1, 2]  # From present time to twice the current age of the universe

# Solve the differential equation for the future
solution_future = solve_ivp(da_dt, t_span_future, [a_0_future], method='RK45')

# Plot the expansion of the Universe from Big Bang to present
plt.plot(solution_past_present.t, solution_past_present.y[0], label='Past and present')
# Plot the predicted future expansion of the Universe
plt.plot(solution_future.t, solution_future.y[0], linestyle='dotted', label='Predicted future')

plt.xlabel('Time (in units of current age of the Universe)')
plt.ylabel('Scale factor a(t) (in units of current size of the Universe)')
plt.title('Expansion of the Universe')
plt.legend()
plt.show()
