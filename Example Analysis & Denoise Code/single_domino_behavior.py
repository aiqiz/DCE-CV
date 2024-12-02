from align_repeats import align
import math
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt


domino_num = 20
general_path = './table & images/table/10d20c1x1000f/'
time, angle = align(domino_num, general_path) # with dimension 10*5*n

w = 7.5 #width, x-dim
h = 48 #height, y-dim
d = 24 #depth, z-dim
s = 1 * w #spacing
theta_until_gravity = round(90 - math.degrees(math.acos(h/(math.sqrt(h*h + w*w)))),4)

fig, axs = plt.subplots(2, 1, figsize=(10, 8))


for i in range(20):
    time_temp = []
    angle_temp = []
    for j in range(len(time[i])):
        axs[0].scatter(time[i][j], angle[i][j], s=2)
        axs[0].set_title('Aligned angular position ')
        axs[0].set_xlabel("time(s)")
        axs[0].set_ylabel("Theta(degree)")
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
    axs[1].plot(derivative_angle_smooth[:,0], derivative_angle_smooth[:,1],label=f'Domino {i+1}', linewidth=2)
    axs[1].set_title('Angular Velocity Curve with Loess fit')
    axs[1].set_xlabel("time(s)")
    axs[1].set_ylabel("angular velocity (degree/s)")


    '''
    df2 = pd.DataFrame(derivative_angle_smooth, columns=["velocity", "time"])
    angular_velocity = np.gradient(df2["velocity"], df2["time"])
    angular_velocity_smooth = sm.nonparametric.lowess(angular_velocity, df["time"], frac=0.1)
    #axs[0, 1].plot(df["time"], derivative_angle, label="Loess Fit")
    axs[2].plot(angular_velocity_smooth[:,0], angular_velocity_smooth[:,1],label=f'Domino {i+1}', linewidth=2)
    axs[2].set_title('Angular Velocity Curve with Loess fit')
    axs[2].set_xlabel("time(s)")
    axs[2].set_ylabel("angular velocity (degree/s)")
    '''
    axs[0].legend()
    axs[1].legend()
    

fig.suptitle("Angular Position and Velocity for 5d10c1x3x1000f 1st-8st Domino", fontsize=20)

plt.tight_layout()
plt.show()