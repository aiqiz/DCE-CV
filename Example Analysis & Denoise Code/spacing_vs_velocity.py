from align_repeats import align
import math
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy import stats
from scipy.signal import medfilt
from scipy.optimize import curve_fit



domino_num = 10
general_path = './table & images/table/5d10c1x420f/'
time, angle = align(domino_num, general_path) # with dimension 10*5*n


fig, axs = plt.subplots(2, 1, figsize=(10, 8))

max_velocity = []

for i in range(domino_num):
    time_temp = []
    angle_temp = []
    for j in range(len(time[i])):
        for elem in time[i][j]:
            time_temp.append(elem)
        for elem in angle[i][j]:
            angle_temp.append(elem)

    


    angle_temp = medfilt(angle_temp, kernel_size=9)
    smoothed = sm.nonparametric.lowess(endog=time_temp, exog=angle_temp, frac=0.15)
    axs[0].plot(smoothed[:,1], smoothed[:, 0], linewidth=2)
    
    df = pd.DataFrame(smoothed, columns=["angle", "time"])
    df = df.round({"time": 10, "angle": 10})
    df = df.drop_duplicates()
    derivative_angle = np.gradient(df["angle"], df["time"])
    derivative_angle_smooth = sm.nonparametric.lowess(derivative_angle, df["time"], frac=0.1)
    #axs[0, 1].plot(df["time"], derivative_angle, label="Loess Fit")
    axs[0].plot(derivative_angle_smooth[:,0], derivative_angle_smooth[:,1], linewidth=2)
    axs[0].set_title('Angular Velocity Curve with Loess fit')
    axs[0].set_xlabel("time(s)")
    axs[0].set_ylabel("angular velocity (degree/s)")

    max_velocity.append(min(derivative_angle_smooth[:,1]))

x = np.linspace(1, domino_num, domino_num)

linear_fit = np.polyfit(x, max_velocity, 1)
linear_y = np.polyval(linear_fit, x)

# Polynomial fit (degree 2)
poly_fit = np.polyfit(x, max_velocity, 2)
poly_y = np.polyval(poly_fit, x)

# Exponential fit
def exponential_func(x, a, b):
    return a * np.exp(b * x)


def compute_chi_square(y_obs, y_pred):
    return np.sum((y_obs - y_pred)**2 / y_pred)

def compute_r_squared(y_obs, y_pred):
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1 - (ss_res / ss_tot)


exp_fit_params, _ = curve_fit(exponential_func, x, max_velocity, p0=(1, 0.1))
exp_y = exponential_func(x, *exp_fit_params)


axs[1].scatter(x, max_velocity, label='max velocity', s = 10)
axs[1].plot(x, linear_y, label='Linear fit', color='blue')
axs[1].plot(x, poly_y, label='Polynomial fit', color='red')
axs[1].plot(x, exp_y, label='Exponential fit', color='green')
axs[1].set_title('max velocity diagram for 20 domino chain')
axs[1].set_xlabel("domino number")
axs[1].set_ylabel("velocity (degree/s)")

print(max_velocity)
# Linear Model Chi-Square and R^2
chi_square_linear = compute_chi_square(max_velocity, linear_y)
r_squared_linear = compute_r_squared(max_velocity, linear_y)

# Polynomial Model Chi-Square and R^2
chi_square_poly = compute_chi_square(max_velocity, poly_y)
r_squared_poly = compute_r_squared(max_velocity, poly_y)

# Exponential Model Chi-Square and R^2
chi_square_exp = compute_chi_square(max_velocity, exp_y)
r_squared_exp = compute_r_squared(max_velocity, exp_y)

# Print results
print(f"Linear Model: Chi-Square = {chi_square_linear:.4f}, R^2 = {r_squared_linear:.4f}")
print(f"Polynomial Model: Chi-Square = {chi_square_poly:.4f}, R^2 = {r_squared_poly:.4f}")
print(f"Exponential Model: Chi-Square = {chi_square_exp:.4f}, R^2 = {r_squared_exp:.4f}")


axs[0].legend()
axs[1].legend()
    

fig.suptitle("Velocity for 10d20c1x1000f 1st-20st Domino", fontsize=20)



plt.tight_layout()
plt.show()