import numpy as np

def euler2quaternion_d(r, p, y):
    # 将输入转换为numpy数组以确保一致性
    r = np.asarray(r)
    p = np.asarray(p)
    y = np.asarray(y)
    
    cr = np.cos(np.deg2rad(r/2))
    cp = np.cos(np.deg2rad(p/2))
    cy = np.cos(np.deg2rad(y/2))
    sr = np.sin(np.deg2rad(r/2))
    sp = np.sin(np.deg2rad(p/2))
    sy = np.sin(np.deg2rad(y/2))
    
    # 检查输入数组长度是否相等
    assert len(r) == len(p) and len(p) == len(y), "Input arrays must have the same length"
    
    # 初始化四元数数组
    q = np.zeros((4, len(r)))
    
    # 计算四元数分量
    q[0,:] = cr*cp*cy + sr*sp*sy
    q[1,:] = sr*cp*cy - cr*sp*sy
    q[2,:] = cr*sp*cy + sr*cp*sy
    q[3,:] = cr*cp*sy - sr*sp*cy
    
    return q