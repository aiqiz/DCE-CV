import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np


general_path = './table & images/table/5d10c0.75x420f/'
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
smoothed_dict = {}

for i in range(1,6):
    df = pd.read_csv(general_path + str(i) + '.csv', names=['domino_num', 'time', 'start_point', 'end_point', 'angle'])
    df = df.apply(pd.to_numeric, errors='coerce')

    for domino in range(1, 2):
        filtered_df = df[df['domino_num'] == domino]
        time = filtered_df['time'].tolist()
        angle = filtered_df['angle'].tolist()
        
        smoothed = sm.nonparametric.lowess(endog=angle, exog=time, frac=0.15)
        smoothed_time = smoothed[:, 0]
        smoothed_angle = smoothed[:, 1]

        derivative_angle = np.gradient(smoothed_angle, smoothed_time)

        smoothed_dict[domino] = smoothed
        
        ax[0].scatter(time, angle, label=f'Original {domino}', s=2)
        ax[0].plot(smoothed_time, smoothed_angle, label=f'Smoothed Repeat {i}')

        ax[1].plot(smoothed_time, derivative_angle, label="Derivative of Curve")



ax[0].set_xlabel('Time')
ax[0].set_ylabel('Angle')
ax[0].set_title('Original and Smoothed Curves for domino_num 0-9')

ax[0].legend()
plt.show()
