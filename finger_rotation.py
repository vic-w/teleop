from scipy.spatial.transform import Rotation as R
import numpy as np

def quaternion_multiply(q1, q2):
    """
    计算四元数乘法 q1 * q2
    使用 * 运算符代替 multiply() 方法
    """
    return R.from_quat(q1) * R.from_quat(q2)

def quaternion_inverse(q):
    """
    计算四元数的逆（共轭四元数）
    """
    return R.from_quat(q).inv().as_quat()

def quaternion_to_axis_angle(q):
    """
    将四元数转换为旋转轴和旋转角度
    :param q: 四元数 (x, y, z, w)
    :return: 旋转轴 (axis), 旋转角度 (angle)
    """
    r = R.from_quat(q)  # 创建 Rotation 对象
    rotvec = r.as_rotvec()  # 获取旋转向量
    axis = rotvec / np.linalg.norm(rotvec)  # 旋转轴是旋转向量的方向
    angle = np.linalg.norm(rotvec)  # 旋转角度是旋转向量的长度
    return axis, angle


def get_relative_rotation(q0, q1):
    """
    计算q0和q1之间的相对旋转（相对旋转轴接近q0的x轴）
    """
    # 计算q0的逆四元数
    q0_inv = quaternion_inverse(q0)
    
    # 计算相对旋转四元数 q_rel = q0_inv * q1
    q_rel = quaternion_multiply(q0_inv, q1)
    
    # 将四元数 q_rel 转换为旋转轴和旋转角度
    axis, angle = quaternion_to_axis_angle(q_rel.as_quat())  # 提取四元数并传递给 R.from_quat
    
    return axis, angle

