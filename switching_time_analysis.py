import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
import h5py



### b defines the position of the switching
def gaussian_error_function(x, a, b, c, d):
    return a * (1 + erf((x - b) / c)) / 2 + d

def gaussian(x, a, b, sigma, c):
    return a * np.exp(-((x - b) ** 2) / (2 * sigma**2)) + c

def gaussian_fit(y, x):
    mean_arg = np.argmin(y)
    mean = x[mean_arg]
    sigma = 5e6
    popt, pcov = curve_fit(
        gaussian,
        x,
        y,
        bounds=(
            (-np.inf, min(x), -np.inf, -np.inf),
            (0, max(x), np.inf, np.inf),
        ),
        p0=[min(y) - max(y), mean, sigma, max(y)],
    )
    return popt

def gaussian_error_function_fit(y, x):
    popt, pcov = curve_fit(
        gaussian_error_function,
        x,
        y,
        bounds=(
            (-15e-4, 25, 0, -10e-4),
            (15e-4, 100, 15, 10e-4),
        ),
        p0=[y[-1] - y[0], 55, 10, y[0]],
    ) 
    return popt


# Reading datafile
file_path = "20231207_163000_pi_pulse_scope_cut.h5"
data_file = h5py.File(file_path)
state = np.array(data_file["data"])
qubit_freq = np.array(data_file["x"])[:, 0]
time_delay = np.array(data_file["y"][0])

# Setting up figure
fig = plt.figure(figsize=(12, 6))
ax1= fig.add_subplot(1,2,1)
ax2= fig.add_subplot(2,2,2)
ax3= fig.add_subplot(2,2,4)

# Finding initial and final frequencies
popt_initial = gaussian_fit(state[:, 0], qubit_freq)
initial_frequency = popt_initial[1]
ax2.scatter(qubit_freq, state[:, 0], 
         label = f"initial f = {initial_frequency/1e6:.1f}MHz")
ax2.plot(qubit_freq, gaussian(qubit_freq, *popt_initial))
popt_final = gaussian_fit(state[:, -1], qubit_freq)
final_frequency = popt_final[1]
ax2.scatter(qubit_freq, state[:, -1],
         label = f"final f = {final_frequency/1e6:.1f}MHz")
ax2.plot(qubit_freq, gaussian(qubit_freq, *popt_final))
ax2.legend()


# Cut 2D plot at initial and final frequencies; fit to gaussian error
initial_f_line = np.argmin(np.abs(qubit_freq-initial_frequency))
final_f_line = np.argmin(np.abs(qubit_freq-final_frequency))

popt_A = gaussian_error_function_fit(state[initial_f_line, :], 
                                     time_delay[:])
ax3.scatter(time_delay, state[initial_f_line, :])
ax3.plot(time_delay, gaussian_error_function(time_delay, *popt_A),
         label = f"Leave initial at {popt_A[1]*4:.1f}ns")

popt_B = gaussian_error_function_fit(state[final_f_line, :], 
                                     time_delay[:])
ax3.scatter(time_delay, state[final_f_line, :])
ax3.plot(time_delay, gaussian_error_function(time_delay, *popt_B),
         label = f"Reach final at {popt_B[1]*4:.1f}ns")

switching_time = popt_B[1]- popt_A[1]
ax3.set_title(f"Switching time of {switching_time*4:.1f}ns")
ax3.legend()

# Plot 2D piscope plot with cuts
X, Y = np.meshgrid(time_delay, qubit_freq)
ax1.pcolormesh(Y, X, state, cmap='viridis')
ax1.plot([qubit_freq[initial_f_line], qubit_freq[initial_f_line]],
         [time_delay[0], time_delay[-1]], linestyle = "--", c = "white")
ax1.plot([qubit_freq[final_f_line], qubit_freq[final_f_line]],
         [time_delay[0], time_delay[-1]], linestyle = "--", c = "white")
ax1.set_ylabel("Wait time (cc)")

print(popt_A, popt_B)
plt.tight_layout()
plt.show()