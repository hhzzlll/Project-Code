import numpy as np

def qExp(q):
    a = q[0]
    v = np.zeros(3)
    v[0] = q[1]
    v[1] = q[2]
    v[2] = q[3]
    
    vn = np.linalg.norm(v)
    
    if vn == 0:
        out = np.zeros(4)
        out[0] = 1
        return out
    else:
        expa = np.exp(a)
        out = np.zeros(4)
        out[0] = expa * np.cos(vn)
        out[1:] = expa * (v/vn) * np.sin(vn)
        return out