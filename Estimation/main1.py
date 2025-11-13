import json
import os
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
import sys
from numpy.typing import NDArray

# 添加工具路径
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, 'utilspy')
sys.path.append(utils_dir)

# 导入工具函数
from utilspy.angle_between_vectors import angle_between_vectors
from utilspy.qInv import qInv
from utilspy.qMul import qMul
from utilspy.transFromQuat import transFromQuat
from estimatePose import estimatePose
from myStateTransitionFcn import myStateTransitionFcn
from myMeasurementLikelihoodFcn import myMeasurementLikelihoodFcn
from types_def import (
    Config,
    Data,
)

def _load_toml(path: Path) -> Dict[str, Any]:
    try:
        import toml  # type: ignore
        return toml.load(str(path))
    except Exception:
        # Python 3.11+ fallback using tomllib
        try:
            import tomllib  # type: ignore
            with open(path, 'rb') as f:
                return tomllib.load(f)
        except Exception as e:
            raise e


def _guess_app_config_path() -> Path | None:
    # 优先 Estimation/../config/app_config.toml
    base = Path(__file__).resolve().parent.parent
    candidates = [
        base / 'config' / 'config.toml',
        # base / 'config' / 'config.toml',
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_config(config_path: str | None = None) -> Config:
    """从配置文件加载参数（优先 TOML，其次 JSON）并转换为 Config。

    支持的 TOML 结构（示例）：
    [project]
    project_name = "20251028"  # 或 project_dir = "d:/.../Data/20251028"

    [estimation.file_paths]
    imu_farm_fpath = "..."
    ...
    """
    # 1) 优先尝试 config.toml
    app_cfg_path = _guess_app_config_path()
    if app_cfg_path and app_cfg_path.suffix.lower() in ('.toml',):
        data = _load_toml(app_cfg_path)
        est = data.get('estimation') if isinstance(data, dict) else None
        prj = data.get('project') if isinstance(data, dict) else None
        if isinstance(est, dict):
            d = est  # 期望包含 file_paths 等五大块
            # 如果存在 project_dir 占位符，则展开
            project_dir: str | None = None
            if isinstance(prj, dict):
                raw_dir = prj.get('project_dir')
                raw_name = prj.get('project_name')
                # 解析 project_dir
                if isinstance(raw_dir, str) and raw_dir.strip():
                    p = Path(raw_dir.strip())
                    if not p.is_absolute():
                        p = (Path(__file__).resolve().parent.parent / p).resolve()
                    project_dir = str(p)
                elif isinstance(raw_name, str) and raw_name.strip():
                    p = (Path(__file__).resolve().parent.parent / 'Data' / raw_name.strip()).resolve()
                    project_dir = str(p)

            # 路径插值：{project_dir}
            if project_dir and isinstance(d.get('file_paths'), dict):
                fp = d['file_paths']
                for k, v in list(fp.items()):
                    if isinstance(v, str) and '{project_dir}' in v:
                        fp[k] = v.replace('{project_dir}', project_dir)

            cfg: Config = Config.from_dict(d)
            cfg.validate()
            return cfg

    # # 2) 其次尝试 JSON（保持兼容）
    # json_path = Path(config_path) if config_path else Path(__file__).parent / 'config.json'
    # if json_path.exists():
    #     with open(json_path, 'r', encoding='utf-8') as f:
    #         cfg_dict = json.load(f)
    #     cfg = Config.from_dict(cfg_dict)
    #     cfg.validate()
    #     return cfg

    raise FileNotFoundError('未找到配置：请在 config/app_config.toml 中提供 estimation.* 配置，或保留 Estimation/config.json')

def state_transition_adapter(state: NDArray[np.float64],
                             angular_velocity: NDArray[np.float64],
                             dt: float,
                             _unused: object) -> NDArray[np.float64]:
    """适配器函数，将estimatePose期望的参数格式转换为myStateTransitionFcn的格式"""
    # 简单的四元数积分，基于角速度
    # 这里实现一个基本的四元数积分方法
    w = angular_velocity
    w_norm = np.linalg.norm(w)
    
    if w_norm < 1e-6:
        return state  # 如果角速度很小，状态不变
    
    # 四元数积分公式
    w_unit = w / w_norm
    angle = w_norm * dt
    
    # 增量四元数
    dq = np.array([
        np.cos(angle/2),
        w_unit[0] * np.sin(angle/2),
        w_unit[1] * np.sin(angle/2), 
        w_unit[2] * np.sin(angle/2)
    ])
    
    # 四元数乘法更新状态
    from utilspy.qMul import qMul
    new_state = qMul(state.reshape(-1, 1), dq.reshape(-1, 1)).flatten()
    
    # 归一化四元数
    new_state = new_state / np.linalg.norm(new_state)
    
    return new_state

def main():
    # 加载配置
    config = load_config()
    
    # 从配置文件中获取参数
    file_paths = config.file_paths
    camera_params = config.camera_parameters
    transformation_matrix = config.transformation_matrix
    initial_quaternion = config.initial_quaternion
    other_params = config.other_parameters
    
    # 提取具体参数
    imu_farm_fpath = file_paths.imu_farm_fpath
    imu_uarm_fpath = file_paths.imu_uarm_fpath
    image_fpath = file_paths.image_fpath
    ground_fpath = file_paths.ground_fpath
    
    fx = camera_params.fx
    fy = camera_params.fy
    
    T_cw = np.array(transformation_matrix.T_cw)
    
    q_ie = np.array(initial_quaternion.q_ie).reshape(-1, 1)
    
    numParticles = other_params.num_particles
    hz_imu = other_params.hz_imu
    hz_image = other_params.hz_image
    idx_sync_w = other_params.idx_sync_w
    idx_sync_kpts = other_params.idx_sync_kpts
    
    # 设置随机种子（模拟 MATLAB 的 rng("default")）
    np.random.seed(0)
    
    print("配置参数已加载:")
    print(f"IMU频率: {hz_imu} Hz")
    print(f"图像频率: {hz_image} Hz")
    print(f"粒子数量: {numParticles}")
    print(f"同步索引 - IMU: {idx_sync_w}, 关键点: {idx_sync_kpts}")
    
    # 加载数据
    print("正在加载数据...")
    try:
        # 使用 pandas 读取 CSV，确保数值列被正确解析
        imu_farm_df = pd.read_csv(imu_farm_fpath)
        imu_uarm_df = pd.read_csv(imu_uarm_fpath)
        image_df = pd.read_csv(image_fpath)
        
        # 去掉包含非数值数据的列（如空格字符）
        # 只保留数值列
        imu_farm_df = imu_farm_df.select_dtypes(include=[np.number])
        imu_uarm_df = imu_uarm_df.select_dtypes(include=[np.number])
        image_df = image_df.select_dtypes(include=[np.number])
        
        # 转换为 numpy 数组，确保数值类型
        imu_farm = imu_farm_df.values.astype(float)
        imu_uarm = imu_uarm_df.values.astype(float)
        image_data = image_df.values.astype(float)
        
    except Exception as e:
        print(f"读取 IMU/图像数据时出错: {e}")
        # 如果类型转换失败，尝试手动处理
        imu_farm_df = pd.read_csv(imu_farm_fpath).select_dtypes(include=[np.number])
        imu_uarm_df = pd.read_csv(imu_uarm_fpath).select_dtypes(include=[np.number])
        image_df = pd.read_csv(image_fpath).select_dtypes(include=[np.number])
        imu_farm = imu_farm_df.values.astype(float)
        imu_uarm = imu_uarm_df.values.astype(float)
        image_data = image_df.values.astype(float)
    
    # 提取关键点数据
    kpts_shoulder_elbow = image_data[:, 1:7]  # [u,v,conf] for shoulder-elbow
    kpts_elbow_wrist = image_data[:, 4:10]    # [u,v,conf] for elbow-wrist
    
    # 提取时间和四元数数据，确保类型正确
    t_imu_farm = imu_farm[:, 1].astype(float)
    quat_farm = imu_farm[:, 2:6].T.astype(float)  # 转置以匹配MATLAB格式
    
    t_imu_uarm = imu_uarm[:, 1].astype(float)
    quat_uarm = imu_uarm[:, 2:6].T.astype(float)  # 转置以匹配MATLAB格式    # 加载真值数据
    try:
        print("正在加载真值数据...")
        motion_df = pd.read_csv(ground_fpath)
        print(f"真值数据形状: {motion_df.shape}")
        print(f"真值数据前几列: {motion_df.columns[:10].tolist()}")
        
        # 尝试将所有列转换为数值类型，跳过非数值列
        motion = motion_df.iloc[2:, :].values  # 跳过前两行
        
        # 确保数据是数值类型
        if motion.dtype == 'object':
            # 如果包含非数值数据，尝试转换
            motion_numeric = []
            for i in range(motion.shape[0]):
                row = []
                for j in range(motion.shape[1]):
                    try:
                        val = float(motion[i, j]) if motion[i, j] != '' else 0.0
                        row.append(val)
                    except (ValueError, TypeError):
                        row.append(0.0)  # 无法转换的值设为0
                motion_numeric.append(row)
            motion = np.array(motion_numeric, dtype=float)
        else:
            motion = motion.astype(float)
            
        print(f"处理后的真值数据形状: {motion.shape}")
        
    except Exception as e:
        print(f"加载真值数据时出错: {e}")
        print("尝试使用替代方法...")
        # 使用numpy直接读取，跳过可能的头部行
        try:
            motion = np.loadtxt(ground_fpath, delimiter=',', skiprows=3)
        except:
            # 最后的备选方案
            motion_df = pd.read_csv(ground_fpath, header=None)
            motion = motion_df.iloc[2:, :].values
            motion = np.array([[float(x) if str(x).replace('.','').replace('-','').isdigit() else 0.0 
                              for x in row] for row in motion], dtype=float)
    
    # 关键点列表和索引（对应MATLAB代码）
    offset = 2
    keyPointList = ['Neck','RShoulder','RElbow','RWrist','LShoulder','LElbow','LWrist','midHip',
                   'RHip','RKnee','RAnkle','LHip','LKnee','LAnkle','LBigToe','LSmallToe','LHeel',
                   'RBigToe','RSmallToe','RHeel']
    idxList = list(range(offset, 60, 3))
    
    # 创建关键点索引字典
    keyPointIndex_XYZ = dict(zip(keyPointList, idxList))
    
    # 计算真值数据
    sz_ground = len(motion)
    uarm_ground = np.full((3, sz_ground), np.nan)
    farm_ground = np.full((3, sz_ground), np.nan)
    angle_ground = np.full(sz_ground, np.nan)
    print("正在计算真值数据...")
    for i in range(sz_ground):
        try:
            # 获取关键点坐标
            idx = keyPointIndex_XYZ['RShoulder']
            rshoulder = motion[i, idx:idx+3].astype(float)
            
            idx = keyPointIndex_XYZ['RElbow']
            relbow = motion[i, idx:idx+3].astype(float)
            
            idx = keyPointIndex_XYZ['RWrist']
            rwrist = motion[i, idx:idx+3].astype(float)
            
            # 计算上臂和前臂向量
            uarm = (relbow - rshoulder)
            farm = (rwrist - relbow)
            
            # 检查向量长度是否有效
            uarm_norm = np.linalg.norm(uarm)
            farm_norm = np.linalg.norm(farm)
            
            if uarm_norm > 1e-6 and farm_norm > 1e-6:
                uarm_ground[:, i] = uarm / uarm_norm
                farm_ground[:, i] = farm / farm_norm
                angle_ground[i] = angle_between_vectors(-uarm, farm)
            else:
                # 如果向量长度太小，设为NaN
                uarm_ground[:, i] = np.nan
                farm_ground[:, i] = np.nan
                angle_ground[i] = np.nan
                
        except Exception as e:
            print(f"处理第{i}行数据时出错: {e}")
            # 设置为NaN以避免程序崩溃
            uarm_ground[:, i] = np.nan
            farm_ground[:, i] = np.nan
            angle_ground[i] = np.nan
            continue
    
    # 同步数据
    print("正在同步数据...")
    t_sync_imu = t_imu_farm[idx_sync_w - 1]  # Python索引从0开始
    f_imu = 1 / hz_imu
    dt_imu = f_imu / (1 / 1000000)  # [1 tick = 1s/1M = 1us]
    
    f_image = 1 / hz_image
    dt_image = f_image / (1 / 1000000)  # [1 tick = 1s/1M = 1us]
    t0 = t_sync_imu - dt_image * (idx_sync_kpts - 1)
    n_image = len(image_data)
    # t_image = np.arange(t0, t0 + n_image * dt_image, dt_image)
    temp_array = np.arange(t0, t0 + n_image * dt_image, dt_image)
    t_image = temp_array[:n_image]
    
    # 准备数据
    idx_imu_start = 0  # Python索引从0开始
    q_se_farm = quat_farm[:, idx_imu_start:idx_imu_start+1]
    q_se_uarm = quat_uarm[:, idx_imu_start:idx_imu_start+1]
    q_es_farm = qInv(q_se_farm)
    q_es_uarm = qInv(q_se_uarm)
    
    q_ei = qInv(q_ie)
    q0_farm = qMul(q_ei, q_se_farm)
    q0_uarm = qMul(q_ei, q_se_uarm)
    
    # 偏差校正
    bias_farm = np.array([0.006147164, -0.029417848, 0.01161752])
    bias_uarm = np.array([0.037155388, 0.017845646, 0.009383009])
    
    # 准备角速度数据
    w_se_farm = np.column_stack([
        t_imu_farm[idx_imu_start:] * 1e-6,
        np.deg2rad(imu_farm[idx_imu_start:, 9:12])  # 角速度列
    ])
    w_se_farm[:, 1] -= bias_farm[0]
    w_se_farm[:, 2] -= bias_farm[1]
    w_se_farm[:, 3] -= bias_farm[2]
    
    w_se_uarm = np.column_stack([
        t_imu_farm[idx_imu_start:] * 1e-6,
        np.deg2rad(imu_uarm[idx_imu_start:, 9:12])  # 角速度列
    ])
    w_se_uarm[:, 1] -= bias_uarm[0]
    w_se_uarm[:, 2] -= bias_uarm[1]
    w_se_uarm[:, 3] -= bias_uarm[2]
    
    # 准备关键点数据
    kpts_uarm = np.column_stack([t_image * 1e-6, kpts_shoulder_elbow])
    kpts_farm = np.column_stack([t_image * 1e-6, kpts_elbow_wrist])    # 姿态估计
    print("正在进行姿态估计...")
    
    # 确保q0是1维数组
    q0_uarm_flat = q0_uarm.flatten()
    q0_farm_flat = q0_farm.flatten()
    # 纯积分（无滤波）
    noFilter = True
    int_uarm = estimatePose(q0_uarm_flat, numParticles, w_se_uarm, kpts_uarm, fx, fy, T_cw, 
                           myStateTransitionFcn, myMeasurementLikelihoodFcn, noFilter)
    int_farm = estimatePose(q0_farm_flat, numParticles, w_se_farm, kpts_farm, fx, fy, T_cw, 
                           myStateTransitionFcn, myMeasurementLikelihoodFcn, noFilter)

    # 使用滤波器
    noFilter = False
    qEst_uarm = estimatePose(q0_uarm_flat, numParticles, w_se_uarm, kpts_uarm, fx, fy, T_cw, 
                            myStateTransitionFcn, myMeasurementLikelihoodFcn, noFilter)
    qEst_farm = estimatePose(q0_farm_flat, numParticles, w_se_farm, kpts_farm, fx, fy, T_cw, 
                            myStateTransitionFcn, myMeasurementLikelihoodFcn, noFilter)

    # 可视化估计结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i in range(2):
        # 前臂IMU
        axes[i, 0].plot(range(qEst_farm.shape[1]), qEst_farm[i, :])
        axes[i, 0].set_title(f'qEst_farm - 维度 {i+1}')
        axes[i, 0].set_xlabel('时间')
        axes[i, 0].set_ylabel('幅值')
        
        # 上臂IMU
        axes[i, 1].plot(range(qEst_uarm.shape[1]), qEst_uarm[i, :])
        axes[i, 1].set_title(f'qEst_uarm - 维度 {i+1}')
        axes[i, 1].set_xlabel('时间')
        axes[i, 1].set_ylabel('幅值')
    
    plt.tight_layout()
    plt.show()
    
    # 重建身体向量
    print("正在重建身体向量...")
    sz_imu = qEst_uarm.shape[1]
    
    farm_int = np.full((3, sz_imu), np.nan)
    uarm_int = np.full((3, sz_imu), np.nan)
    farm_est = np.full((3, sz_imu), np.nan)
    uarm_est = np.full((3, sz_imu), np.nan)
    farm_imu = np.full((3, sz_imu), np.nan)
    uarm_imu = np.full((3, sz_imu), np.nan)

    # 旋转矩阵序列与 IMU 四元数轨迹缓存
    R_uarm_int = np.full((sz_imu, 3, 3), np.nan)
    R_farm_int = np.full((sz_imu, 3, 3), np.nan)
    R_uarm_est = np.full((sz_imu, 3, 3), np.nan)
    R_farm_est = np.full((sz_imu, 3, 3), np.nan)
    R_uarm_imu = np.full((sz_imu, 3, 3), np.nan)
    R_farm_imu = np.full((sz_imu, 3, 3), np.nan)

    q_imu_uarm = np.full((4, sz_imu), np.nan)
    q_imu_farm = np.full((4, sz_imu), np.nan)
    
    angle_int = np.full(sz_imu, np.nan)
    angle_est = np.full(sz_imu, np.nan)
    angle_imu = np.full(sz_imu, np.nan)
    
    quat_record = []
    lb = np.array([[-1], [0], [0]])  # 局部身体向量
    
    for i in range(sz_imu):
        # 积分结果
        T_se = transFromQuat(int_uarm[:, i:i+1])
        T_es = T_se.T
        R_uarm_int[i, :, :] = T_se
        uarm_int[:, i] = (T_es @ lb).flatten()
        
        T_se = transFromQuat(int_farm[:, i:i+1])
        T_es = T_se.T
        R_farm_int[i, :, :] = T_se
        farm_int[:, i] = (T_es @ lb).flatten()
        
        angle_int[i] = angle_between_vectors(-uarm_int[:, i], farm_int[:, i])
        
        # 估计结果
        T_se = transFromQuat(qEst_uarm[:, i:i+1])
        T_es = T_se.T
        R_uarm_est[i, :, :] = T_se
        uarm_est[:, i] = (T_es @ lb).flatten()
        
        T_se = transFromQuat(qEst_farm[:, i:i+1])
        T_es = T_se.T
        R_farm_est[i, :, :] = T_se
        farm_est[:, i] = (T_es @ lb).flatten()
        
        angle_est[i] = angle_between_vectors(-uarm_est[:, i], farm_est[:, i])
        
        # IMU直接结果
        qimu_uarm_i = qMul(q_ei, quat_uarm[:, idx_imu_start+i:idx_imu_start+i+1])
        T_se = transFromQuat(qimu_uarm_i)
        T_es = T_se.T
        q_imu_uarm[:, i] = qimu_uarm_i.flatten()
        R_uarm_imu[i, :, :] = T_se
        uarm_imu[:, i] = (T_es @ lb).flatten()
        
        qimu_farm_i = qMul(q_ei, quat_farm[:, idx_imu_start+i:idx_imu_start+i+1])
        T_se = transFromQuat(qimu_farm_i)
        quat_record.append(qimu_farm_i)
        T_es = T_se.T
        q_imu_farm[:, i] = qimu_farm_i.flatten()
        R_farm_imu[i, :, :] = T_se
        farm_imu[:, i] = (T_es @ lb).flatten()
        
        angle_imu[i] = angle_between_vectors(-uarm_imu[:, i], farm_imu[:, i])
    
    # 绘制前臂xyz结果
    plt.figure(figsize=(12, 10))
    for i in range(3):
        plt.subplot(3, 1, i+1)
        if i == 0:
            plt.title(f"粒子数: {numParticles}")
        
        time_axis = (t_imu_farm[idx_imu_start:idx_imu_start+sz_imu] - t_imu_farm[idx_imu_start]) * 1e-6
        plt.plot(time_axis, farm_imu[i, :], 'r-', label='Xsens DOT')
        plt.plot(time_axis, farm_est[i, :], 'b-', label='Est')
        
        # 真值数据
        if sz_ground <= len(t_image):
            time_ground = (t_image[:sz_ground] - t_imu_farm[idx_imu_start]) * 1e-6
            plt.plot(time_ground, farm_ground[i, :sz_ground], 'g-', label='Ground Truth')
        
        plt.ylabel(chr(ord('x') + i))
        plt.legend()
        plt.grid(True, alpha=0.3)
    
        plt.xlabel("时间 [s]")
        plt.tight_layout()
        plt.show()
    
    # 计算角度误差
    print("正在计算角度误差...")
    err_angle_int = np.full(sz_ground, np.nan)
    err_angle_imu = np.full(sz_ground, np.nan)
    err_angle_est = np.full(sz_ground, np.nan)
    debug_int = np.full(sz_ground, np.nan)
    debug_imu = np.full(sz_ground, np.nan)
    debug_est = np.full(sz_ground, np.nan)
    debug_ground = np.full(sz_ground, np.nan)
    
    for i in range(sz_ground):
        t = t_image[i]
        
        # 找到最近的IMU时间
        i_imu = np.where(t_imu_farm <= t)[0]
        if len(i_imu) > 0:
            idx_imu = i_imu[-1] - idx_imu_start - 1
            if 0 <= idx_imu < sz_imu:
                err_angle_int[i] = angle_int[idx_imu] - angle_ground[i]
                err_angle_imu[i] = angle_imu[idx_imu] - angle_ground[i]
                err_angle_est[i] = angle_est[idx_imu] - angle_ground[i]
                
                debug_ground[i] = angle_ground[i]
                debug_int[i] = angle_int[idx_imu]
                debug_imu[i] = angle_imu[idx_imu]
                debug_est[i] = angle_est[idx_imu]
    
    # 绘制调试图
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    valid_indices = ~np.isnan(debug_int)
    plt.plot(np.where(valid_indices)[0], debug_int[valid_indices], 'b-', linewidth=1.5)
    plt.xlabel('时间')
    plt.ylabel('debug_int')
    plt.title('Int')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    valid_indices = ~np.isnan(debug_imu)
    plt.plot(np.where(valid_indices)[0], debug_imu[valid_indices], 'r-', linewidth=1.5)
    plt.xlabel('时间')
    plt.ylabel('debug_imu')
    plt.title('IMU')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    valid_indices = ~np.isnan(debug_est)
    plt.plot(np.where(valid_indices)[0], debug_est[valid_indices], 'g-', linewidth=1.5)
    plt.xlabel('时间')
    plt.ylabel('debug_est')
    plt.title('Est')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 计算并显示统计结果
    valid_est = ~np.isnan(err_angle_est)
    valid_imu = ~np.isnan(err_angle_imu)
    valid_int = ~np.isnan(err_angle_int)
    
    if np.any(valid_est):
        est_mean = np.mean(np.rad2deg(np.abs(err_angle_est[valid_est])))
        est_std = np.std(np.rad2deg(np.abs(err_angle_est[valid_est])))
        print(f"[Est] 平均误差: {est_mean:.2f}°, 标准差: {est_std:.2f}°")
    
    if np.any(valid_imu):
        imu_mean = np.mean(np.rad2deg(np.abs(err_angle_imu[valid_imu])))
        imu_std = np.std(np.rad2deg(np.abs(err_angle_imu[valid_imu])))
        print(f"[IMU] 平均误差: {imu_mean:.2f}°, 标准差: {imu_std:.2f}°")
    
    if np.any(valid_int):
        int_mean = np.mean(np.rad2deg(np.abs(err_angle_int[valid_int])))
        int_std = np.std(np.rad2deg(np.abs(err_angle_int[valid_int])))
        print(f"[Int] 平均误差: {int_mean:.2f}°, 标准差: {int_std:.2f}°")
    
    # 汇总为 Data 结构
    data = Data(
        # 时间
        t_imu_us=t_imu_farm[idx_imu_start:idx_imu_start+sz_imu],
        t_image_us=t_image,
        # 四元数轨迹
        int_uarm=int_uarm,
        int_farm=int_farm,
        est_uarm=qEst_uarm,
        est_farm=qEst_farm,
        imu_uarm=q_imu_uarm,
        imu_farm=q_imu_farm,
        # 旋转矩阵（T_se，传感器->地球）
        R_uarm_int=R_uarm_int,
        R_farm_int=R_farm_int,
        R_uarm_est=R_uarm_est,
        R_farm_est=R_farm_est,
        R_uarm_imu=R_uarm_imu,
        R_farm_imu=R_farm_imu,
        # 单位向量与角度
        uarm_int=uarm_int,
        farm_int=farm_int,
        uarm_est=uarm_est,
        farm_est=farm_est,
        uarm_imu=uarm_imu,
        farm_imu=farm_imu,
        angle_int=angle_int,
        angle_est=angle_est,
        angle_imu=angle_imu,
        # 真值
        angle_ground=angle_ground,
        uarm_ground=uarm_ground,
        farm_ground=farm_ground,
        # 观测
        w_se_uarm=w_se_uarm,
        w_se_farm=w_se_farm,
        kpts_uarm=kpts_uarm,
        kpts_farm=kpts_farm,
        # 标定
        fx=fx,
        fy=fy,
        T_cw=T_cw,
        q_ie=q_ie,
        q_ei=q_ei,
        # 其他
        meta={
            "numParticles": numParticles,
            "hz_imu": hz_imu,
            "hz_image": hz_image,
            "idx_sync_w": idx_sync_w,
            "idx_sync_kpts": idx_sync_kpts,
        }
    )
    data.ensure_seconds()
    print(f"Data 已组装：T={sz_imu}, 图像帧={len(t_image)}")

    print("处理完成！")

if __name__ == "__main__":
    main()
