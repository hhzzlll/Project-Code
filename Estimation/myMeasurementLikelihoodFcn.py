import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

def myMeasurementLikelihoodFcn(x_pred, y, fx, fy, T_ce):
    """
    x_pred: shape (4,) or (4, 3000) - quaternion(s)
    y: measurement vector
    fx, fy: focal lengths
    T_ce: transformation matrix
    """    
    pt0 = y[0:2]  # Python 0-based indexing
    pt1 = y[3:5]  # MATLAB y(4:5) -> Python y[3:5]
    
    d = pt1 - pt0
    du = d[0]
    dv = d[1]
    
    n = 10000
    sigma = 5
    Gx = sigma * np.random.randn(n)
    Gy = sigma * np.random.randn(n)
    samples = (du + Gx) / (dv + Gy)
    
    # Handle different shapes of x_pred
    if x_pred.ndim == 1:  # shape (4,)
        x_pred = x_pred.reshape(4, 1)
    
    # Convert quaternion to rotation and apply
    # MATLAB quatrotate(quatconj(q), v) equivalent
    num_quaternions = x_pred.shape[1]
    l_w_correct = np.zeros((3, num_quaternions))
    
    for i in range(num_quaternions):
        # x_pred is [w, x, y, z] format, scipy expects [x, y, z, w]
        quat_scipy = np.array([x_pred[1, i], x_pred[2, i], x_pred[3, i], x_pred[0, i]])
        rot = R.from_quat(quat_scipy)
        
        # Apply rotation to [1, 0, 0]
        l_w_t = rot.apply([1, 0, 0])
        l_w = l_w_t  # Already a column vector equivalent
        
        # Camera world reference frame transformation
        l_w_correct[0, i] = -l_w[2]  # -l_w(3,:) in MATLAB
        l_w_correct[1, i] = -l_w[1]  # -l_w(2,:) in MATLAB
        l_w_correct[2, i] = -l_w[0]  # -l_w(1,:) in MATLAB
    
    # Camera frame
    l_c = T_ce @ l_w_correct
    
    # Project onto pixel frame
    du_pred = fx * l_c[0, :]
    dv_pred = fy * l_c[1, :]
    
    x = du_pred / dv_pred
    
    # Histogram-based likelihood calculation
    binWidth = 0.05
    binWidthHalf = binWidth / 2

    # 处理异常值，确保 samples 是有限的
    samples_finite = samples[np.isfinite(samples)]
    if len(samples_finite) == 0:
        return np.ones_like(x) if hasattr(x, '__len__') else 1.0
    
    # 计算合理的范围（类似MATLAB自动处理）
    sample_min = np.floor(samples_finite.min() / binWidth) * binWidth
    sample_max = np.ceil(samples_finite.max() / binWidth) * binWidth
    
    # 创建bin边界，直接指定bin宽度
    bins = np.arange(sample_min, sample_max + binWidth, binWidth)
    
    # 如果bins太多，限制范围
    if len(bins) > 2000:  # 限制最大bins数量
        # 使用百分位数限制范围
        p1, p99 = np.percentile(samples_finite, [1, 99])
        sample_min = np.floor(p1 / binWidth) * binWidth
        sample_max = np.ceil(p99 / binWidth) * binWidth
        bins = np.arange(sample_min, sample_max + binWidth, binWidth)
    
    # 计算直方图
    P, edges = np.histogram(samples_finite, bins=bins)
    
    # 计算bin中心 - 对应MATLAB的 edge+binWidthHalf
    bin_centers = edges[:-1] + binWidthHalf
    
    # 插值 - 对应MATLAB的 interp1(...,'spline')
    if len(bin_centers) > 1 and len(P) > 1:
        # 确保x在合理范围内
        x_clipped = np.clip(x, bin_centers.min(), bin_centers.max())
        f = interp1d(bin_centers, P / n, kind='cubic', fill_value=0, bounds_error=False)
        likelihood = np.maximum(0, f(x_clipped))
    else:
        likelihood = np.zeros_like(x)
    
    return likelihood
    # P, edges = np.histogram(samples, bins=int((samples.max() - samples.min()) / binWidth))
    # bin_centers = edges[:-1] + binWidthHalf
    
    # # Interpolation with extrapolation handling
    # if len(bin_centers) > 1:
    #     f = interp1d(bin_centers, P / n, kind='cubic', fill_value=0, bounds_error=False)
    #     likelihood = np.maximum(0, f(x))
    # else:
    #     likelihood = np.zeros_like(x)
    
    # return likelihood