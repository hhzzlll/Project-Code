import numpy as np

def myStateTransitionFcn(x_prev, w, dt, n):
    """
    四元数状态转移函数 - 优化版本
    
    参数:
    - x_prev: 四元数状态 (4×1或4×n矩阵)
    - w: 角速度 (1×3矩阵)
    - dt: 时间步长 (标量)
    - n: 样本数量 (1或3000)
    
    返回:
    - x_pred: 预测的四元数状态
    """
    # 生成带噪声的角速度
    wn = w - 0.0025 * np.random.randn(n, 3)
    
    # 向量化的quatexp计算
    half_wn_dt = 0.5 * wn * dt
    theta = np.linalg.norm(half_wn_dt, axis=1, keepdims=True)
    
    # 初始化为单位四元数
    dq = np.zeros((n, 4))
    dq[:, 0] = 1.0
    
    # 只处理非零角度的情况
    nonzero = theta.flatten() > 1e-10
    if np.any(nonzero):
        dq[nonzero, 0] = np.cos(theta[nonzero].flatten())
        sin_theta_div_theta = np.sin(theta[nonzero].flatten()) / theta[nonzero].flatten()
        dq[nonzero, 1:] = half_wn_dt[nonzero] * sin_theta_div_theta[:, None]
    
    # 处理输入四元数的形状
    if len(x_prev.shape) == 1:
        # 如果x_prev是一维数组，将其重塑为列向量
        x_prev = x_prev.reshape(-1, 1)
    
    if x_prev.shape[1] == 1 and n > 1:
        # 如果x_prev是单列而n>1，则复制扩展
        q1 = np.tile(x_prev.T, (n, 1))
    else:
        q1 = x_prev.T
    
    # 向量化的quatmultiply计算
    q2 = dq
    
    # 批量四元数乘法 (不需要循环)
    x_pred_trans = np.zeros((n, 4))
    
    # 提取四元数分量
    a1, b1, c1, d1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    a2, b2, c2, d2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    # 四元数乘法公式 (向量化)
    x_pred_trans[:, 0] = a1*a2 - b1*b2 - c1*c2 - d1*d2
    x_pred_trans[:, 1] = a1*b2 + b1*a2 + c1*d2 - d1*c2
    x_pred_trans[:, 2] = a1*c2 - b1*d2 + c1*a2 + d1*b2
    x_pred_trans[:, 3] = a1*d2 + b1*c2 - c1*b2 + d1*a2
    
    # 转置回原始形状
    x_pred = x_pred_trans.T
    
    # 向量化归一化
    norm = np.sqrt(np.sum(x_pred**2, axis=0, keepdims=True))
    x_pred = x_pred / norm

    if n == 1:
        return x_pred.flatten()
    return x_pred