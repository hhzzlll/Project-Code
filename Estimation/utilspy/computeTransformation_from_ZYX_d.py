import numpy as np

def computeTransformation_from_ZYX_d(y, p, r):
    # y: yaw [deg]
    # p: pitch [deg]
    # r: roll [deg]
    
    # Convert trig functions to work with degrees
    cy = np.cos(np.deg2rad(y))
    cp = np.cos(np.deg2rad(p))
    cr = np.cos(np.deg2rad(r))

    sy = np.sin(np.deg2rad(y))
    sp = np.sin(np.deg2rad(p))
    sr = np.sin(np.deg2rad(r))

    # Create transformation matrix
    out = np.array([
        [cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],
        [cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],
        [-sp,   sr*cp,          cr*cp]
    ])
    
    return out