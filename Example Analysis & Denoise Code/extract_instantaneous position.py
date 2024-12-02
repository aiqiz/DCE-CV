from align_repeats import align
import math
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt


domino_num = 10
general_path = './table & images/table/5d10c3x1000f/'
time, angle = align(domino_num, general_path) # with dimension 10*5*n

w = 7.5 #width, x-dim
h = 48 #height, y-dim
d = 24 #depth, z-dim
s = 1 * w #spacing
theta_until_gravity = round(90 - math.degrees(math.acos(h/(math.sqrt(h*h + w*w)))),4)

fig, axs = plt.subplots(2, 2, figsize=(9, 7))


for i in range(4):
    time_temp = []
    angle_temp = []
    for j in range(len(time[i])):
        axs[0, 0].scatter(time[i][j], angle[i][j], s=2)
        axs[0, 0].set_title('Aligned angular position ')
        axs[0, 0].set_xlabel("time(s)")
        axs[0, 0].set_ylabel("Theta(degree)")

        axs[1, 0].scatter(time[i][j], angle[i][j], s=2)
        axs[1, 0].set_title('Aligned Angular Position with Loess Fit')
        axs[1, 0].set_xlabel("time(s)")
        axs[1, 0].set_ylabel("Theta(degree)")
        for elem in time[i][j]:
            time_temp.append(elem)
        for elem in angle[i][j]:
            angle_temp.append(elem)


    smoothed = sm.nonparametric.lowess(endog=time_temp, exog=angle_temp, frac=0.15)
    axs[1,0].plot(smoothed[:,1], smoothed[:, 0], linewidth=2)
    
    df = pd.DataFrame(smoothed, columns=["angle", "time"])
    df = df.round({"time": 10, "angle": 10})
    df = df.drop_duplicates()
    derivative_angle = np.gradient(df["angle"], df["time"])
    derivative_angle_smooth = sm.nonparametric.lowess(derivative_angle, df["time"], frac=0.1)
    #axs[0, 1].plot(df["time"], derivative_angle, label="Loess Fit")
    axs[0, 1].plot(derivative_angle_smooth[:,0], derivative_angle_smooth[:,1],label=f'Domino {i+1}', linewidth=2)
    axs[0, 1].set_title('Angular Velocity Curve with Loess fit')
    axs[0, 1].set_xlabel("time(s)")
    axs[0, 1].set_ylabel("angular velocity (degree/s)")

    df = pd.DataFrame(smoothed, columns=["angle", "time"])
    df = df.round({"time": 10, "angle": 10})
    df = df.drop_duplicates()
    derivative_angle = np.gradient(df["angle"], df["time"])
    #axs[0, 1].plot(df["time"], derivative_angle, label="Loess Fit")
    axs[1, 1].plot(df['time'], derivative_angle, linewidth=2)
    axs[1, 1].plot(derivative_angle_smooth[:,0], derivative_angle_smooth[:,1],linewidth=2)
    axs[1, 1].set_title('Angular Velocity Curve')
    axs[1, 1].set_xlabel("time(s)")
    axs[1, 1].set_ylabel("angular velocity (degree/s)")

    axs[0,0].legend()
    axs[0,1].legend()
    axs[1,0].legend()
    axs[1,1].legend()

fig.suptitle("Angular Position and First Gradient for 5d10c3x1000f 1st-4st Domino", fontsize=20)

plt.tight_layout()
plt.show()