#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import math

from math import pi, cos, sin
from mpl_toolkits.mplot3d import Axes3D

# Performs matrix multiplication to transform vector
def transform(T: np.array, v: np.array) -> np.array:
    v = np.vstack((v, [1]))
    v_trans = T @ v 
    return np.round(v_trans[:3], decimals=2)    

# Returns line equations in symbolic form
def get_line(p1: np.array, p2: np.array):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    if dx == 0:
        return sp.Eq(x, p1[0])
    
    m = dy / dx
    k = p1[1] - m * p1[0]
    return sp.simplify(m * x + k)

# Linear interpolation between two points
def get_line_points(p1, p2, num=100):
    return np.linspace(p1, p2, num=num).T

if __name__ == "__main__":

    ### Parameters ####
    tc_0 = np.array([0, 0, 1]).reshape(3, 1)
    p1_c = np.array([0.366, 1, 1.36]).reshape(3, 1)
    p2_c = np.array([-0.134, 1, 2.2321]).reshape(3, 1)
    p3_c = np.array([0.366, -1, 1.366]).reshape(3, 1)

    theta = 2*pi/3

    nz = np.array([0, 0, 1])
    ny = np.array([0, 1, 0])
    ###################

    q = np.hstack( (np.array([cos(theta/2)]), ny*sin(theta/2)) )
    qr = q[0]
    qi = q[1]
    qj = q[2]
    qk = q[3]

    ### Homogenous Transformation Matrix ###
    # Quaternion --> Rotation matrix
    R = np.array([[1-2*(qj**2+qk**2), 2*(qi*qj-qk*qr), 2*(qi*qk+qj*qr)],
                [2*(qi*qj+qk*qr), 1-2*(qi**2+qk**2), 2*(qi*qk-qi*qr)],
                [2*(qi*qk-qj*qr), 2*(qj*qk+qi*qr), 1-2*(qi**2+qj**2)]])

    print("\nThe rotation matrix:")
    for row in R:
        print("[" + "  ".join(f"{val:8.4f}" for val in row) + "]")

    # Transformation matrix
    Tc_0 = np.block([ [R,           tc_0],
                    [np.zeros((1, 3)), np.array([[1]])] ])

    print("\nThe homogenous transformation matrix:")
    for row in Tc_0:
        print("[" + "  ".join(f"{val:8.4f}" for val in row) + "]")
    ########################################

    ### Point Transformations ###
    p1_0 = transform(Tc_0, p1_c)
    p2_0 = transform(Tc_0, p2_c)
    p3_0 = transform(Tc_0, p3_c)
    #############################

    ### Lines ###
    x = sp.symbols('x')
    y = sp.symbols('y')

    y1 = get_line(p1_0, p2_0)
    y2 = get_line(p2_0, p3_0)

    print("\nLine p1-p2:  "+f'y1 = {round(y1[0])}')
    print("Line p2-p3:  "+f'y1 = {y2[0]}')
    ##############

    ### Graphical overview ###
    # Transformed points in frame 0 (flattened to 1D)
    points = {
        'P1': p1_0.flatten(),
        'P2': p2_0.flatten(),
        'P3': p3_0.flatten(),
        'Camera': np.array([0, 0, 1])
    }

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    colors = {'P1': 'red', 'P2': 'green', 'P3': 'blue', 'Camera': 'black'}
    markers = {'P1': 'o', 'P2': 'o', 'P3': 'o', 'Camera': 'v'}

    for label, point in points.items():
        ax.scatter(*point, color=colors[label], marker=markers[label], label=label, s=60)
        ax.text(*point, f'  {label}', fontsize=9)

    # Plot lines P1–P2 and P2–P3
    for a, b in [('P1', 'P2'), ('P2', 'P3')]:
        line = get_line_points(points[a], points[b])
        ax.plot(line[0], line[1], line[2], label=f'{a}–{b}', linewidth=2)

    # Plot camera's local coordinate axes
    camera_pos = points['Camera']
    axis_length = 0.3 

    # Local camera axes from rotation matrix R
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    # Draw arrows for each camera axis (X: red, Y: green, Z: blue)
    ax.quiver(*camera_pos, *x_axis, length=axis_length, color='red',   arrow_length_ratio=0.2, linewidth=2, label='Cam x-axis')
    ax.quiver(*camera_pos, *y_axis, length=axis_length, color='green', arrow_length_ratio=0.2, linewidth=2, label='Cam y-axis')
    ax.quiver(*camera_pos, *z_axis, length=axis_length, color='blue',  arrow_length_ratio=0.2, linewidth=2, label='Cam z-axis')

    # Final plot settings
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Points, Lines & Camera Local Coordinate Axes (Frame 0)")
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    ##########################
