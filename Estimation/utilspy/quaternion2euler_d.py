import numpy as np

def quaternion2euler_d(q):
    # Check input dimension
    m = q.shape[0]
    assert m == 4
    
    q0 = q[0, :]
    q1 = q[1, :]
    q2 = q[2, :]
    q3 = q[3, :]
    
    # Calculate Euler angles in degrees
    r = np.rad2deg(np.arctan2(2*q2*q3 + 2*q0*q1, 2*q0**2 + 2*q3**2 - 1))
    p = -np.rad2deg(np.arcsin(2*q1*q3 - 2*q0*q2))
    y = np.rad2deg(np.arctan2(2*q1*q2 + 2*q0*q3, 2*q0**2 + 2*q1**2 - 1))
    
    # Transpose the results
    return r.T, p.T, y.T