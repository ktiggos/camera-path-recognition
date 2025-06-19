#!/usr/bin/env python

from arv_lib import *

if __name__ == "__main__":

    # Flags
    # Apply the new roation of question (c) or not
    new_rot = False

    # Parameters
    tc_0 = np.array([0, 0, 1]).reshape(3, 1)
    f = 0.001
    rho_w = 10**(-5)
    rho_h = 10**(-5)
    u0, v0 = [640, 480]

    if new_rot:
        theta = pi
    else:
        theta = 2*pi/3

    # From part A - path_recognition.py
    p1_0 = np.array([1, 1, 0]).reshape(3, 1)
    p2_0 = np.array([2, 1, 0]).reshape(3, 1)
    p3_0 = np.array([1, -1, 0]).reshape(3, 1)

    if new_rot:
        Tc_0 = np.array([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]])
    else:
        Tc_0 = np.array([[-0.5, 0, sqrt(3)/2, 0],
                         [0, 1, 0, 0],
                         [-sqrt(3)/2, 0, -0.5, 1],
                         [0, 0, 0, 1]])
    
    ##### Calculate Camera Points #####
    # Camera matrix
    C1 = np.array([[f/rho_w, 0, u0],
                   [0, f/rho_h, v0],
                   [0, 0, 1]])
    C2 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    
    C = C1 @ C2

    # Camera points transformations
    c1 = camera_transform(C, inv(Tc_0), p1_0, rdec = 0)
    c2 = camera_transform(C, inv(Tc_0), p2_0, rdec = 0)
    c3 = camera_transform(C, inv(Tc_0), p3_0, rdec = 0)

    print(f'\nCamera point 1: {c1} (px)')
    print(f'Camera point 2: {c2} (px)')
    print(f'Camera point 3: {c3} (px)\n')
    ###################################

    ##### FOV Condition Check #####
    # FOV calculation
    FOV = [2*atan2(2*u0*rho_w, 2*f),
           2*atan2(2*v0*rho_h, 2*f)] # (theta_h, theta_v)
    
    # Points relative to camera frame
    p1_c = transform(inv(Tc_0), p1_0, rdec = 3)
    p2_c = transform(inv(Tc_0), p2_0, rdec = 3)
    p3_c = transform(inv(Tc_0), p3_0, rdec = 3)

    print("FOV check:")
    if inFOV(p1_c, FOV):
        print(f'Point p1_c : IN FOV')
    else:
        print(f'Point p1_c : OUT OF FOV')

    if inFOV(p2_c, FOV):
        print(f'Point p2_c : IN FOV')
    else:
        print(f'Point p2_c : OUT OF FOV')

    if inFOV(p3_c, FOV):
        print(f'Point p3_c : IN FOV')
    else:
        print(f'Point p3_c : OUT OF FOV')
    print("\n")