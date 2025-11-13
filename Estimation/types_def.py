from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray

# ---------- 常用数组类型别名（仅用于类型提示，不影响运行） ----------
Vec3 = NDArray[np.float64]             # shape = (3,)
Quat = NDArray[np.float64]             # shape = (4,) 或 (4,1)
QuatT = NDArray[np.float64]            # shape = (4, T)
Mat3 = NDArray[np.float64]             # shape = (3,3)
TimeVec = NDArray[np.float64]          # shape = (N,)（单位见上下文）

# IMU 角速度时间序列 [t(sec), wx, wy, wz]
IMUOmegaSeries = NDArray[np.float64]   # shape = (N, 4)

# 关键点观测时间序列 [t(sec), u1, v1, c1, u2, v2, c2]
KptsSeries = NDArray[np.float64]       # shape = (M, 7)

# 3D 向量轨迹（单位方向）
Vec3T = NDArray[np.float64]            # shape = (3, T)

# 角度序列（弧度）
AngleT = NDArray[np.float64]           # shape = (T,)


# ---------- 配置数据结构 ----------
@dataclass
class FilePaths:
    imu_farm_fpath: str
    imu_uarm_fpath: str
    image_fpath: str
    ground_fpath: str

@dataclass
class CameraParameters:
    fx: float
    fy: float

@dataclass
class TransformationMatrix:
    # 建议配置中为 4x4 或 3x3/3x4 的嵌套列表；代码中再转 np.array
    T_cw: List[List[float]]

@dataclass
class InitialQuaternion:
    # 四元数顺序约定为 [w, x, y, z]
    q_ie: List[float]  # len == 4

@dataclass
class OtherParameters:
    num_particles: int
    hz_imu: float
    hz_image: float
    idx_sync_w: int      # 1-based
    idx_sync_kpts: int   # 1-based

@dataclass
class Config:
    file_paths: FilePaths
    camera_parameters: CameraParameters
    transformation_matrix: TransformationMatrix
    initial_quaternion: InitialQuaternion
    other_parameters: OtherParameters

    @staticmethod
    def from_dict(d: dict) -> "Config":
        return Config(
            file_paths=FilePaths(**d["file_paths"]),
            camera_parameters=CameraParameters(**d["camera_parameters"]),
            transformation_matrix=TransformationMatrix(**d["transformation_matrix"]),
            initial_quaternion=InitialQuaternion(**d["initial_quaternion"]),
            other_parameters=OtherParameters(**d["other_parameters"]),
        )

    def validate(self) -> None:
        # 基础字段校验
        if len(self.initial_quaternion.q_ie) != 4:
            raise ValueError("initial_quaternion.q_ie 长度必须为 4（[w,x,y,z]）")
        if self.camera_parameters.fx <= 0 or self.camera_parameters.fy <= 0:
            raise ValueError("camera_parameters.fx/fy 必须为正数")
        # 同步索引 >=1
        if self.other_parameters.idx_sync_w < 1 or self.other_parameters.idx_sync_kpts < 1:
            raise ValueError("idx_sync_w / idx_sync_kpts 必须为 1-based 且 >= 1")

        # T_cw 是矩阵嵌套列表的最基本检查
        if not isinstance(self.transformation_matrix.T_cw, list) or not self.transformation_matrix.T_cw:
            raise ValueError("transformation_matrix.T_cw 必须为嵌套列表")


# ---------- 算法结果聚合（可选） ----------
@dataclass
class PoseEstimates:
    # 四元数轨迹
    int_uarm: QuatT
    int_farm: QuatT
    qEst_uarm: QuatT
    qEst_farm: QuatT

@dataclass
class ReconstructedVectors:
    # 机体轴在地球坐标下的单位向量轨迹
    uarm_int: Vec3T
    farm_int: Vec3T
    uarm_est: Vec3T
    farm_est: Vec3T
    uarm_imu: Vec3T
    farm_imu: Vec3T

    angle_int: AngleT
    angle_est: AngleT
    angle_imu: AngleT

@dataclass
class GroundTruth:
    uarm_ground: Vec3T
    farm_ground: Vec3T
    angle_ground: AngleT


# ---------- Data：运行过程数据聚合 ----------
@dataclass
class Data:
    """
    用于组织姿态估计流程中的核心数据：
    - 时间轴（IMU/图像，微秒与秒）
    - 四元数轨迹（积分/估计/IMU直接）
    - 旋转矩阵序列（与四元数对应）
    - 机体轴单位向量与关节角
    - 观测（角速度与关键点）与标定信息

    形状约定：
    - 时间向量: (N,)
    - 四元数轨迹: (4, T) 约定分量顺序 [w, x, y, z]
    - 旋转矩阵序列: (T, 3, 3)
    - 单位向量轨迹: (3, T)
    - 角速度时间序列: (N, 4) = [t(sec), wx, wy, wz]
    - 关键点时间序列: (M, 7) = [t(sec), u1, v1, c1, u2, v2, c2]
    """

    # 时间戳
    t_imu_us: Optional[TimeVec] = None
    t_imu_s: Optional[TimeVec] = None
    t_image_us: Optional[TimeVec] = None
    t_image_s: Optional[TimeVec] = None

    # 四元数轨迹 (4, T)
    int_uarm: Optional[QuatT] = None
    int_farm: Optional[QuatT] = None
    est_uarm: Optional[QuatT] = None
    est_farm: Optional[QuatT] = None
    imu_uarm: Optional[QuatT] = None
    imu_farm: Optional[QuatT] = None

    # 旋转矩阵序列 (T, 3, 3)
    R_uarm_int: Optional[np.ndarray] = None
    R_farm_int: Optional[np.ndarray] = None
    R_uarm_est: Optional[np.ndarray] = None
    R_farm_est: Optional[np.ndarray] = None
    R_uarm_imu: Optional[np.ndarray] = None
    R_farm_imu: Optional[np.ndarray] = None

    # 单位向量 (3, T)
    uarm_int: Optional[Vec3T] = None
    farm_int: Optional[Vec3T] = None
    uarm_est: Optional[Vec3T] = None
    farm_est: Optional[Vec3T] = None
    uarm_imu: Optional[Vec3T] = None
    farm_imu: Optional[Vec3T] = None

    # 角度 (T,)
    angle_int: Optional[AngleT] = None
    angle_est: Optional[AngleT] = None
    angle_imu: Optional[AngleT] = None

    # 真值（与图像时间对齐）
    angle_ground: Optional[AngleT] = None
    uarm_ground: Optional[Vec3T] = None
    farm_ground: Optional[Vec3T] = None

    # 观测：角速度与关键点
    w_se_uarm: Optional[IMUOmegaSeries] = None
    w_se_farm: Optional[IMUOmegaSeries] = None
    kpts_uarm: Optional[KptsSeries] = None
    kpts_farm: Optional[KptsSeries] = None

    # 标定/初始量
    fx: Optional[float] = None
    fy: Optional[float] = None
    T_cw: Optional[np.ndarray] = None  # (3x3 | 4x4)
    q_ie: Optional[Quat] = None        # (4,1) or (4,)
    q_ei: Optional[Quat] = None        # (4,1) or (4,)

    # 其他元信息
    meta: Dict[str, Any] = field(default_factory=dict)

    def ensure_seconds(self) -> None:
        """如果只有微秒时间，则自动生成秒时间（原地填充）。"""
        if self.t_imu_s is None and self.t_imu_us is not None:
            self.t_imu_s = self.t_imu_us.astype(np.float64) * 1e-6
        if self.t_image_s is None and self.t_image_us is not None:
            self.t_image_s = self.t_image_us.astype(np.float64) * 1e-6
