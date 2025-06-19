#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import math

from math import pi, sqrt, cos, sin, tan, atan2
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import inv

x = sp.symbols('x')
y = sp.symbols('y')

# Performs matrix multiplication to transform vector
def transform(T: np.array, v: np.array, rflag: bool = True, rdec: np.uint = 2) -> np.array:
    v = np.vstack((v, [1]))
    v_trans = T @ v

    if rflag:
        return np.round(v_trans[:3], decimals=rdec)
    else:
        return v_trans[:3]  

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

# Transform 3D vector to 2D camera coordinates
def camera_transform(C: np.array, T: np.array, v: np.array, rflag: bool = True, rdec: np.uint = 2) -> np.array:
    v = np.vstack((v, [1]))
    c_ = C @ T @ v
    c_ = c_[:3,0]
    c = np.array([c_[0]/c_[2], c_[1]/c_[2]])

    if rflag:
        return np.round(c, decimals=rdec)
    else:
        return c
    
def inFOV(p: np.array, FOV: list) -> bool:
    wp = 2 * p[-1] * tan(FOV[0]/2)
    hp = 2 * p[-1] * tan(FOV[1]/2)

    if (p[0] <= wp/2) and (p[1] <= hp/2):
        return True
    else:
        return False 