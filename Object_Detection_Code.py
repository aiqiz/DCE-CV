import cv2
from collections import Counter
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt


def swap(x,y):
    k = x
    x = y
    y = k
    return x, y

def top_frequencies(list, top_n, uncertainty_range):
    adjusted_frequencies = Counter()
    current_value = list[0]
    current_count = 1
    for value in list[1:]:
        if abs(value - current_value) <= uncertainty_range:
            current_count += 1
        else:
            adjusted_frequencies[current_value] += current_count
            current_value = value
            current_count = 1
    
    adjusted_frequencies[current_value] += current_count
    top_frequencies = adjusted_frequencies.most_common(top_n)
    top_frequencies = [x[0] for x in top_frequencies]
    return top_frequencies

def initial_x(pos, domino_num, uncertainty_range):
    x = []
    for row in pos:
        x.append(row[0])
        x.append(row[2])
    x.sort()
    initial_position_x = top_frequencies(x, domino_num*2, uncertainty_range)
    initial_position_x.sort()
    return initial_position_x
    
def initial_y(pos, uncertainty_range):
    y = []
    for row in pos:
        y.append(row[1])
        y.append(row[3])
    y.sort()
    initial_position_x = top_frequencies(y, 1, uncertainty_range)
    return initial_position_x[0]

def intersection_and_angle(x1, y1, x2, y2, c):
    '''
    return the intersection point coord and angle degree
    '''
    # height and width for the bottom line of the screen
    k = (y2 - y1) / (x2 - x1)
    b = y1 - x1*k
    x = (c-b)/k
    angle_radians = math.atan(k)
    angle_degrees = math.degrees(angle_radians)
    return x, abs(angle_degrees)

def instantaneous_drawing(x1, y1, x2, y2, initial_x, initial_y, x_starting):
    if x1 < x_starting or x2 < x_starting:
        return None, None, None, None, None
    if (y1 > y2 and x2 > x2) or (y2 > y1 and x1 > x2):
        return None, None, None, None, None
    if x1 == x2:
        return None, None, None, None, None
    if y1 > y2 and x1 < x2:
        x1, x2 = swap(x1, x2)
        y1, y2 = swap(y1, y2)  

    end_x, angle = intersection_and_angle(x1, y1, x2, y2, initial_y)
    start_x = x1
    start_y = y1
    end_x = end_x
    end_y = initial_y
    return start_x, start_y, end_x, end_y, angle

def determin_which_domino(endx, initial_x, uncertainty):
    closest = None
    min_diff = float('inf')

    for i in range(len(initial_x)):
        if abs(initial_x[i] - endx) <= uncertainty:
            diff = abs(initial_x[i] - endx)
            if diff < min_diff:
                min_diff = diff
                closest = i
    return closest


for repeat in range(2,4):
    frame_num = 0
    initial_pos = []
    initial_base = []
    domino_num = 10
    frame_speed = 1000 #per second
    general_directory = './Video Data - processed /5d10c0.75x420f/'
    image_directory = './table & images/image/'
    table_directory = './table & images/table/'
    directory = general_directory + str(repeat) + '.mp4'
    cap = cv2.VideoCapture(directory)      
    #object_detector = cv2.createBackgroundSubtractorMOG2(history = 50, detectShadows=False, varThreshold = 30)
    k = 0
    dic_pos = []

    while True:
        frame_num = frame_num + 1
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape # 720*1080
        #mask = object_detector.apply(frame)
        #mask1 = d1.apply(frame)
        #mask2 = d2.apply(frame)
        #mask3 = d3.apply(frame)
        roi = frame
        #480*640
        
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(roi,(3, 3),0)
        low_threshold = 50
        high_threshold = 200
        L2gradient = False
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold, L2gradient=L2gradient)

        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 10  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 30 # minimum number of pixels making up a line
        max_line_gap = 8  # maximum gap in pixels between connectable line segments
        line_image = np.copy(frame) * 0  # creating a blank to draw lines on

        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
        
        if type(lines) == np.ndarray:
            for line in lines:
                for x1,y1,x2,y2 in line:
                    if frame_num <= 200:
                        cv2.line(roi, (x1,y1), (x2,y2), (255,0,0), 2)
                        if abs(x1-x2) <= 5:
                            initial_pos.append([x1, y1, x2, y2])
                        if abs(y1-y2) <= 2:
                            initial_base.append([x1, y1, x2, y2])
                    else:    
                        if abs(y1-y2) >= 2:
                            cv2.line(roi, (x1 ,y1), (x2, y2), (0,0,255), 2)
                            cv2.line(roi, (0,initial_baseline_y), (width,initial_baseline_y), (0,255,0), 1) #draw baseline
                            for x in initial_position_x:
                                cv2.line(roi, (x,100), (x,1000), (0,255,0), 1) #draw initial position

                            startx, starty, endx, endy, angle = instantaneous_drawing(x1, y1, x2, y2, initial_position_x, initial_baseline_y, x_starting)
                            if angle != 90  and angle != None and angle > 10: 
                                k = k+1
                                domino_num = determin_which_domino(endx, initial_position_x, uncertainty=4)
                                if domino_num != None:
                                    cv2.line(roi, (startx ,starty), (int(endx), endy), (0,0,255), 2)
                                    dic_pos.append([domino_num, frame_num/frame_speed, (startx, starty), (int(endx), endy), angle])
                            

            if frame_num == 200:
                width_by_pixel = 0
                initial_position_x = initial_x(initial_pos, domino_num, uncertainty_range=4) # get initial x pos of dommino
                ini_pos_x = initial_position_x
                for i in range(len(initial_position_x)):
                    if i%2==0:
                        width_by_pixel = width_by_pixel + initial_position_x[i+1] - initial_position_x[i]
                width_by_pixel = width_by_pixel/domino_num
                height_by_pixel = int((width_by_pixel/9 )*40)
                initial_baseline_y = initial_y(initial_base, uncertainty_range=2)-6 # get initial pos of baseline
                cv2.line(roi, (0,initial_baseline_y), (width,initial_baseline_y), (0,255,0), 1) # draw baseline
                ini_pos = []
                len_x = len(initial_position_x)
                x_starting = initial_position_x[0]
                for i in range(2*domino_num):
                    if i%2 == 1:
                        ini_pos.append(initial_position_x[i])
                print(len(ini_pos))
                for x in ini_pos:
                    cv2.line(roi, (x,100), (x,1000), (0,255,0), 1)
                initial_position_x = ini_pos
                print(initial_baseline_y, initial_position_x)

        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) 
        if key == 27:
            break


    dic_pos.sort(key=lambda x: x[0])
    print(k)
    print(frame_num)
    cap.release()
    cv2.destroyAllWindows()

    position_table = pd.DataFrame(dic_pos, columns=['domino_num', 'time', 'start_point', 'end_point', 'angle'])
    
    cap = cv2.VideoCapture(directory)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 4)
    ret, frame = cap.read()


    for i in range(len(dic_pos)):
        (x1, y1) = dic_pos[i][3]
        x2 = x1 + math.cos(math.radians(dic_pos[i][4]))*height_by_pixel
        y2 = y1 - math.sin(math.radians(dic_pos[i][4]))*height_by_pixel
        cv2.circle(frame, radius=2, center=(int(x2), int(y2)), color=(0,0,255))

    for x in ini_pos:
        cv2.line(frame, (x,100), (x,1000), (0,255,0), 1) #draw initial position
    cv2.line(frame, (0,initial_baseline_y), (width,initial_baseline_y), (0,255,0), 1) #draw baseline

    cv2.imshow('Last Frame', frame)
    cv2.imwrite(image_directory + str(repeat) + '.png', frame)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


    position_table.to_csv(table_directory + str(repeat) + '.csv')