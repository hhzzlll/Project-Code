import numpy as np
import computeTransformation_from_ZYX_d
import transFromQuat

def computeTransformationMatrix_d(data, type):
    """
    Args:
        data: numpy array with shape (n, 3) for euler angles [roll, pitch, yaw] in degrees
              or (n, 4) for quaternions
        type: string, either "euler" or "quaternion"
    Returns:
        list of transformation matrices
    """
    n = data.shape[0]
    out = []

    if type == "euler":
        for i in range(n):
            y = data[i, 2]  # yaw
            p = data[i, 1]  # pitch
            r = data[i, 0]  # roll
            out.append(computeTransformation_from_ZYX_d(y, p, r))
            
    elif type == "quaternion":
        for i in range(n):
            q = data[i, :]
            out.append(transFromQuat(q))
            
    else:
        print("Warning: unknown type")
        
    return out
