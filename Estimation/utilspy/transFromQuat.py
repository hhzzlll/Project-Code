import numpy as np

def transFromQuat(q):
    """
    Convert quaternion to rotation matrix
    Args:
        q: quaternion (shape (4,) or (4,1))
    Returns:
        T: 3x3 rotation matrix
    """
    q = np.array(q).flatten()  # Ensure 1D array
    
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    
    T = np.array([
        [1-2*y**2-2*z**2, 2*(x*y+w*z), 2*(x*z-w*y)],
        [2*(x*y-w*z), 1-2*x**2-2*z**2, 2*(y*z+w*x)],
        [2*(x*z+w*y), 2*(y*z-w*x), 1-2*x**2-2*y**2]
    ])
    
    return T


if __name__ == "__main__":
    # Example quaternion
    q_example = np.array([0.58315975, -0.579921967, -0.405531363, 0.3989383])  # Example quaternion [w, x, y, z]

    # Convert to rotation matrix
    rotation_matrix = transFromQuat(q_example)
    
    print("Rotation Matrix from Quaternion:")
    print(rotation_matrix)
    
    