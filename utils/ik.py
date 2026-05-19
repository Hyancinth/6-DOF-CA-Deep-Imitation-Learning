"""
Inverse Kinematic Utils for UR5 6 DOF Robotic Arm
"""
import numpy as np
N_JOINTS = 6

def compute_jacobian(transformation_matrices):
    """
    Compute the Jacobian matrix for the end-effector given the transformation matrices of each joint
    """
    pos_ee = transformation_matrices[-1][:3, 3] # Position of the end-effector

    jacobian = np.zeros((6, N_JOINTS)) # Initialize the Jacobian matrix

    for i in range(N_JOINTS):
        z_i = transformation_matrices[i][:3, 2] # Z-axis of the i-th joint frame
        p_i = transformation_matrices[i][:3, 3] # Position of the i-th joint frame

        jacobian[:3, i] = np.cross(z_i, pos_ee - p_i) # Linear velocity
        jacobian[3:, i] = z_i # Angular velocity
    
    return jacobian

def compute_cartesian_velocity(jacobian, joint_velocities):
    """
    Compute the Cartesian velocity of the end-effector given the Jacobian and joint velocities
    """
    return jacobian @ joint_velocities

