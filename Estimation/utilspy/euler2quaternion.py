import numpy as np

def euler2quaternion(r, p, y):
    """
    Convert euler angles to quaternion.
    Args:
        r: roll angle(s)
        p: pitch angle(s)
        y: yaw angle(s)
    Returns:
        q: quaternion array of shape (4, n)
    """
    cr = np.cos(r/2)
    cp = np.cos(p/2)
    cy = np.cos(y/2)
    sr = np.sin(r/2)
    sp = np.sin(p/2)
    sy = np.sin(y/2)
    
    # Check input dimensions
    assert len(r) == len(p) and len(p) == len(y), "Input arrays must have same length"
    
    # Initialize quaternion array
    q = np.zeros((4, len(r)))
    
    # Calculate quaternion components
    q[0,:] = cr*cp*cy + sr*sp*sy
    q[1,:] = sr*cp*cy - cr*sp*sy
    q[2,:] = cr*sp*cy + sr*cp*sy
    q[3,:] = cr*cp*sy - sr*sp*cy
    
    return q