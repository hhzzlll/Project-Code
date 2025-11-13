import numpy as np
from scipy import stats
from utilspy.quaternion2euler_d import quaternion2euler_d
from utilspy.euler2quaternion_d import euler2quaternion_d
import time
from filterpy.monte_carlo import systematic_resample
from filterpy.common import Q_discrete_white_noise
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
# from pfilter import ParticleFilter as PF  # 新增：导入pfilter库

def estimatePose(q0, N, w_sw, keypoints, fx, fy, T_cw, stateTransitionFcn, measurementLikelihoodFcn, noFilter):
    """
    Parameters:
    q0: initial quaternion from world to sensor (q_sw), dimension 3x1
    N: number of particles
    w_sw: timestamped angular velocity of the sensor, dimension T1x4 [rad/s]
    keypoints: timestamped keypoints pixel coordinate, dimension T2x7
    w must be recorded ahead of keypoints
    Returns: estimated quaternion from world to sensor
    """
      # Setup particle filter
    # class ParticleFilter:
    #     def __init__(self, state_transition_fcn, measurement_likelihood_fcn):
    #         self.state_transition_fcn = state_transition_fcn
    #         self.measurement_likelihood_fcn = measurement_likelihood_fcn
    #         self.particles = None
    #         self.weights = None
            
    #     def initialize(self, N, q0, cov):
    #         # 确保q0是1维数组
    #         if q0.ndim > 1:
    #             q0 = q0.flatten()
    #         self.particles = np.random.multivariate_normal(q0, cov, N).T
    #         self.weights = np.ones(N) / N
            
    #     def predict(self, w, dt, N):
    #         # 向量化状态转移，避免循环
    #         if hasattr(self.state_transition_fcn, '__vectorized__'):
    #             self.particles = self.state_transition_fcn(self.particles, w, dt, N)
    #         else:
    #             # 如果函数不支持向量化，尝试批处理
    #             try:
    #                 self.particles = self.state_transition_fcn(self.particles.T, w, dt, 1).T
    #             except:
    #                 # 回退到循环方式
    #                 for i in range(N):
    #                     self.particles[:, i] = self.state_transition_fcn(self.particles[:, i], w, dt, 1)
    #         return np.mean(self.particles, axis=1)
            
    #     def correct(self, measurement, fx, fy, T_cw):
    #         # 计算权重
    #         weights = self.measurement_likelihood_fcn(self.particles, measurement, fx, fy, T_cw)
    #         self.weights = weights / np.sum(weights)
            
    #         # 使用更高效的系统重采样
    #         N = len(self.weights)
    #         indices = systematic_resample(self.weights)
    #         self.particles = self.particles[:, indices]
            
    #         # 直接设置均匀权重，避免数组操作
    #         self.weights.fill(1.0 / N)
            
    #         return np.mean(self.particles, axis=1)
    # class ParticleFilter:
    #     def __init__(self, state_transition_fcn, measurement_likelihood_fcn):
    #         self.state_transition_fcn = state_transition_fcn
    #         self.measurement_likelihood_fcn = measurement_likelihood_fcn
    #         self.particles = None
    #         self.weights = None
    #         # self.n_jobs = multiprocessing.cpu_count()
            
    #     def initialize(self, N, q0, cov):
    #         q0 = np.asarray(q0, dtype=np.float64)
    #         if q0.ndim > 1:
    #             q0 = q0.flatten()
    #         self.particles = np.random.multivariate_normal(q0, cov, N).T
    #         self.weights = np.ones(N, dtype=np.float64) / N

    #     def _state_transition_single(self, particle, w, dt):
    #         """单个粒子的状态转移"""
    #         return self.state_transition_fcn(particle, w, dt, 1)
        
        

    #     def predict(self, w, dt, N):
    #         if hasattr(self.state_transition_fcn, '__vectorized__'):
    #             self.particles = self.state_transition_fcn(self.particles, w, dt, N)
    #         else:
    #             try:
    #                 self.particles = self.state_transition_fcn(self.particles.T, w, dt, 1).T
    #             except:
    #                 # 使用向量化操作替代Numba
    #                 new_particles = np.empty_like(self.particles)
    #                 for i in range(N):
    #                     new_particles[:, i] = self._state_transition_single(self.particles[:, i], w, dt)
    #                 self.particles = new_particles
            
    #         return np.mean(self.particles, axis=1)

    #     def systematic_resample(self, weights):
    #         """优化的系统重采样实现"""
    #         N = len(weights)
    #         # 生成均匀间隔的位置
    #         positions = (np.random.random() + np.arange(N)) / N
    #         indices = np.zeros(N, dtype=np.int64)
    #         cumsum = np.cumsum(weights)
    #         i, j = 0, 0
            
    #         # 查找采样位置
    #         while i < N and j < N:
    #             if positions[i] < cumsum[j]:
    #                 indices[i] = j
    #                 i += 1
    #             else:
    #                 j += 1
                    
    #         return indices

    #     def _measurement_likelihood_batch(self, measurement, fx, fy, T_cw):
    #         """批量计算测量似然"""
    #         N = self.particles.shape[1]
    #         weights = np.empty(N, dtype=np.float64)
    #         for i in range(N):
    #             weights[i] = self.measurement_likelihood_fcn(
    #                 self.particles[:, i], measurement, fx, fy, T_cw
    #             )
    #         return weights

    #     def correct(self, measurement, fx, fy, T_cw):
    #         # 计算权重
    #         if hasattr(self.measurement_likelihood_fcn, '__vectorized__'):
    #             weights = self.measurement_likelihood_fcn(self.particles, measurement, fx, fy, T_cw)
    #         else:
    #             weights = self._measurement_likelihood_batch(measurement, fx, fy, T_cw)
            
    #         # 归一化权重
    #         weights_sum = np.sum(weights)
    #         if weights_sum > 0:
    #             self.weights = weights / weights_sum
    #         else:
    #             self.weights = np.ones_like(weights) / len(weights)

    #         # 重采样
    #         indices = self.systematic_resample(self.weights)
    #         self.particles = self.particles[:, indices]
            
    #         # 重置权重
    #         self.weights = np.ones_like(self.weights) / len(self.weights)
            
    #         return np.mean(self.particles, axis=1)
    class ParticleFilter():
        def __init__(self, stateTransitionFcn, measurementLikelihoodFcn):
            # 初始化父类，粒子初始化等由外部完成
            self.stateTransitionFcn = stateTransitionFcn
            self.measurementLikelihoodFcn = measurementLikelihoodFcn
            # pfilter的初始化需要指定状态转移和观测函数，但我们这里直接用自定义函数
            # 其余参数由外部设置
            self.particles = None
            self.weights = None

        def initialize(self, N, q0, cov):
            q0 = np.asarray(q0, dtype=np.float64)
            if q0.ndim > 1:
                q0 = q0.flatten()
            self.particles = np.random.multivariate_normal(q0, cov, N).T
            self.weights = np.ones(N, dtype=np.float64) / N

        def predict(self, *args, **kwargs):
            # 自动传递self.particles作为第一个参数
            self.particles = self.stateTransitionFcn(self.particles, *args, **kwargs)
            state_estimate = np.sum(self.weights * self.particles, axis=1)
            return state_estimate

        def correct(self, *args, **kwargs):
            # 自动传递self.particles作为第一个参数
            # return self.measurementLikelihoodFcn(self.particles, *args, **kwargs)  
            likelihoods = self.measurementLikelihoodFcn(self.particles, *args, **kwargs)
    
            # 确保似然是1维数组
            if likelihoods.ndim > 1:
                likelihoods = likelihoods.flatten()
            
            # 更新权重：权重 = 先验权重 × 似然
            self.weights = self.weights * likelihoods
            
            # 归一化权重，避免除零错误
            weights_sum = np.sum(self.weights)
            if weights_sum > 1e-15:  # 避免数值下溢
                self.weights = self.weights / weights_sum
            else:
                # 如果权重和太小，重新初始化为均匀分布
                self.weights = np.ones_like(self.weights) / len(self.weights)
            
            # 计算有效样本大小，决定是否重采样
            n_eff = 1.0 / np.sum(self.weights ** 2)
            N = len(self.weights)
            
            # 当有效样本大小小于阈值时进行重采样
            if n_eff < N / 2:
                # 系统重采样
                indices = systematic_resample(self.weights)
                self.particles = self.particles[:, indices]
                # 重采样后重置权重为均匀分布
                self.weights = np.ones(N, dtype=np.float64) / N
            
            # 状态估计：加权平均
            if hasattr(self, 'state_estimation_method') and self.state_estimation_method == 'mean':
                state_estimate = np.average(self.particles, weights=self.weights, axis=1)
            else:
                # 默认使用简单平均
                state_estimate = np.mean(self.particles, axis=1)
            
            return state_estimate 
        
    # Initialize particle filter
    myPF = ParticleFilter(stateTransitionFcn, measurementLikelihoodFcn)
    myPF.initialize(N, q0, np.diag([0.05, 0.05, 0.05, 0.05]))
    # 设置状态估计方法和重采样方法
    myPF.state_estimation_method = 'mean'
    # Generate random initial particles
    # 确保q0是列向量形状以匹配quaternion2euler_d的期望
    q0_reshaped = q0.reshape(-1, 1) if q0.ndim == 1 else q0
    rowd, pitchd, yawd = quaternion2euler_d(q0_reshaped)  # This function needs to be defined
    rowd_rnd = rowd + 3 * np.random.randn(N)
    pitchd_rnd = pitchd + 3 * np.random.randn(N)
    yawd_rnd = yawd + 3 * np.random.randn(N)
    q_rnd = -euler2quaternion_d(rowd_rnd, pitchd_rnd, yawd_rnd)  # This function needs to be defined
    myPF.particles = q_rnd
    
    sz_w = len(w_sw)
    sz_kpts = len(keypoints)
    qEst = np.full((4, sz_w), np.nan)
    dt_w = w_sw[1, 0] - w_sw[0, 0]
    dt_kpts = keypoints[1, 0] - keypoints[0, 0]
    idx_kpts = 0
    
    if noFilter:
        qEst[:, 0] = q0
        for idx_w in range(1, sz_w):
            qEst[:, idx_w] = stateTransitionFcn(qEst[:, idx_w-1], w_sw[idx_w, 1:], dt_w, 1)
        return qEst
    
    # Estimate offline
    process_time = np.full(len(keypoints), np.nan)
    
    for idx_w in range(sz_w):
        if idx_w%1000 == 0:
            print(f"Processing {idx_w}/{sz_w} frames")
        t = w_sw[idx_w, 0]
        
        if idx_kpts < sz_kpts and t - keypoints[idx_kpts, 0] > 0:
            # Has measurement and can detect joints
            assert t - keypoints[idx_kpts, 0] < dt_kpts  # Safety check
            
            # Check if measurement is valid
            if not np.any(keypoints[idx_kpts, 1:] == 0):
                start_time = time.time()
                qEst[:, idx_w] = myPF.correct(keypoints[idx_kpts, 1:], fx, fy, T_cw)
                myPF.predict(w_sw[idx_w, 1:], dt_w, N)
                process_time[idx_kpts] = time.time() - start_time
                idx_kpts += 1
            else:
                # Invalid measurement
                qEst[:, idx_w] = myPF.predict(w_sw[idx_w, 1:], dt_w, N)
                idx_kpts += 1
        else:
            # No measurement
            qEst[:, idx_w] = myPF.predict(w_sw[idx_w, 1:], dt_w, N)
    
    print(f"Average processing time\n mean: {np.nanmean(process_time):.3f}s, std:{np.nanstd(process_time):.3f}")
    return qEst