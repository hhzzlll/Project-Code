import numpy as np

def qMul(p, q):
    """
    Quaternion multiplication (MATLAB compatible)
    Args:
        p: first quaternion (numpy array of shape (4,) or (4,1))
        q: second quaternion (numpy array of shape (4,) or (4,1))
    Returns:
        z: result quaternion (same shape as input)
    """
    # Convert to column vectors if needed
    p_orig_shape = p.shape
    q_orig_shape = q.shape
    
    p = np.array(p).flatten()
    q = np.array(q).flatten()
    
    # Extract components from q
    aq, bq, cq, dq = q[0], q[1], q[2], q[3]
    
    # Create multiplication matrix A
    A = np.array([
        [aq, -bq, -cq, -dq],
        [bq,  aq,  dq, -cq],
        [cq, -dq,  aq,  bq],
        [dq,  cq, -bq,  aq]
    ])
    
    # Compute quaternion multiplication
    z = np.dot(A, p)
    
    # Return in original shape format (prefer column vector for MATLAB compatibility)
    if len(p_orig_shape) == 2 or len(q_orig_shape) == 2:
        return z.reshape(-1, 1)
    else:
        return z