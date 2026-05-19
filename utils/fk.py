"""
Forward Kinematic Utils for UR5 6 DOF Robotic Arm

DH Parameters taken from here: 
https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
"""

import numpy as np

A = [0, -0.425, -0.39225, 0, 0, 0] # link length
D = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823] # link offset
ALPHA = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0] # link twist

# UR5_DH = np.array([A, D, ALPHA])

def compute_dh_transform_matrix(a, d, alpha, theta):
    """
    Compute the homogeneous transformation matrix using DH parameters
    """
    T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                  [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                  [0,              np.sin(alpha),                np.cos(alpha),               d],
                  [0,              0,                            0,                           1]])
    return T


def compute_transformation_matrices(theta):
    """
    Compute the transformation matrices for each joint given the joint angles
    """
    frames = []
    for i in range(6):
        T = compute_dh_transform_matrix(A[i], D[i], ALPHA[i], theta[i])
        frames.append(T)
    return frames


def compute_fk_all_frames(theta):
    """
    Compute the transformation matrices for all frames (base to end-effector)
    """
    transform_matrices = compute_transformation_matrices(theta)
    fk_transform_matrices = [np.eye(4)] # Start with the base frame (identity matrix)
    current_transform = np.eye(4)
    for T in transform_matrices:
        current_transform = current_transform @ T # Multiply the current transformation matrix with the next frame's transformation matrix
        fk_transform_matrices.append(current_transform)
    return fk_transform_matrices

def get_fk_end_effector(theta):
    """
    Get the transformation matrix of the end-effector given the joint angles
    """
    fk_frames = compute_fk_all_frames(theta)

    return fk_frames[-1] # The last frame is the end-effector frame

def get_ee_position(theta):
    """
    Get the position vector of the end-effector given the joint angles
    """
    fk_end_effector = get_fk_end_effector(theta)
    ee_position = fk_end_effector[:3, 3] # Extract the position from the transformation matrix 

    return ee_position

def get_ee_rotation(theta):
    """
    Get the rotation matrix of the end-effector given the joint angles
    """
    fk_end_effector = get_fk_end_effector(theta)
    ee_rotation = fk_end_effector[:3, :3] # Extract the rotation from the transformation matrix

    return ee_rotation

def get_ee_pose(theta):
    """
    Get the pose of the end-effector given the joint angles
    """
    ee_position = get_ee_position(theta) 
    ee_rotation = get_ee_rotation(theta)

    return ee_position, ee_rotation

def get_joint_positions(theta):
    """
    Get the positions of all joints given the joint angles
    """
    fk_frames = compute_fk_all_frames(theta)
    joint_positions = [frame[:3, 3] for frame in fk_frames] # Extract the position of each joint from the transformation matrices

    return joint_positions