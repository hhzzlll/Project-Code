import numpy as np

def qInv(q):
    q_inv = -q.copy()  # 复制并取负
    q_inv[0] = q[0]    # 第一个元素保持不变
    return q_inv