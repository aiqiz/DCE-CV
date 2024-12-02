import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
import copy


def find_region(float_list, p, q):
    start_index = None
    end_index = None

    for i, value in enumerate(float_list):
        if start_index is None and value <= p and float_list[i+1] <= p:
            start_index = i  # First element greater than or equal to p
        if value < q and float_list[i-1] < q and float_list[i-2] < q and q!=0:
            end_index = i - 1  # Last element less than or equal to q
            break
    
    if end_index is None:  # In case q is larger than all elements in the list
        end_index = len(float_list) - 1
        
    return start_index, end_index

def shift_by_index(arr, shift):
    return arr + shift

def calculate_dispersion(shift, ref_x, x):
    shifted_x = shift_by_index(x, shift)  # Shift x-values by index
    shifted_x = np.asarray(shifted_x).flatten()
    ref_x = np.asarray(ref_x).flatten()
    if len(ref_x) > len(shifted_x):
        ref_x = ref_x[len(ref_x)-len(shifted_x):]
    elif len(ref_x) < len(shifted_x):
        shifted_x = shifted_x[len(shifted_x)-len(ref_x):]
    return np.sum((ref_x - shifted_x) ** 2)

def find_optimal_horizontal_shift(ref_x, x):
    # Bound the shift to a smaller range to prevent large shifts
    bounds = [(-1, 1)]
    result = minimize(calculate_dispersion, 0, args=(ref_x, x), method='Powell', bounds=bounds, tol=1e-4)
    return result.x[0]

def align(domino_num, general_path):
    array_time = []
    array_angle = []
    #fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for k in range(domino_num):
        angle = []
        time = []
        for i in range(1, 6):
            df = pd.read_csv(general_path + str(i) + '.csv', names=['domino_num', 'time', 'start_point', 'end_point', 'angle'])
            df = df.apply(pd.to_numeric, errors='coerce')
            temp = df[df["domino_num"] == k]
            time.append(np.array(temp["time"].tolist()))
            angle.append(np.array(temp["angle"].tolist()))
        array_time.append(time)
        array_angle.append(angle)
        '''
        if k == domino_examine:
            for j in range(len(time)):
                axs[0, 0].scatter(time[j], angle[j], label=f'Data {j+1}', s=2)
            axs[0, 0].set_title('Original Data (Different Horizontal Shifts)')
            axs[0, 0].legend()
            '''

    for i in range(domino_num):
        minimum_time = 10000
        x_data = copy.deepcopy(array_time[i])
        y_data = copy.deepcopy(array_angle[i])

        for j in range(0, len(x_data)):
            index_start, index_end = find_region(y_data[j], 85, 75)
            y_data[j] = y_data[j][index_start: index_end]
            x_data[j]= x_data[j][index_start: index_end]

        ref_x = x_data[0]
        shifts = [0]  # No shift for the reference list

        for j in range(1, len(array_time[i])):
            optimal_shift = find_optimal_horizontal_shift(ref_x, x_data[j])
            shifts.append(optimal_shift)
            array_time[i][j] = array_time[i][j] + optimal_shift
            #print(f"Optimal shift for Data {j+1}: {optimal_shift} indices")
            
        
        for j in range(0, len(array_time[i])):
            if min(array_time[i][j]) < minimum_time:
                minimum_time = min(array_time[i][j])
        for j in range(len(array_time[0])):
            array_time[i][j] =  array_time[i][j] - minimum_time
            #index_start, index_end = find_region(array_angle[i][j], 90, 0)
            #array_time[i][j] = array_time[i][j][index_start:index_end]
            #array_angle[i][j] = array_angle[i][j][index_start:index_end]
            
        '''
        for j in range(len(array_time[i])):
            axs[0, 1].scatter(array_time[i][j], array_angle[i][j], label=f'Data {j+1}', s=2)
        axs[0, 1].set_title('Shifted Data (Aligned by Index Shift)')
        axs[0, 1].legend()
        '''

    return array_time, array_angle

