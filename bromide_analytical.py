import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt

# 参数定义
v = 8.2e-5   # m/s
D = 3.6e-7    # m²/s
L = 0.1       # m

# 解析解计算函数（修复除零错误）
def ade_solution(t, x=L):
    if t <= 0:
        return 0.0  # 处理t=0的情况
    eta = (x - v * t) / (2 * np.sqrt(D * t))
    return 0.5 * erfc(eta)

# 实验数据时间点（秒）
t_exp = np.array([5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,175,180]) * 60
C_exp = np.array([0,0.0148,0.1759,0.4855,0.7076,0.8352,0.9142,0.9514,0.9681,0.9768,0.9815,0.9841,0.9862,0.9882,0.9879,0.9899,0.9905,0.9881,0.9908,0.9930,0.9913,0.9887,0.8612,0.3751,0.1979,0.1168,0.0776,0.0516,0.0341,0.0239,0.0167,0.0130,0.0111,0.0072])

# 生成时间范围时跳过t=0（关键修改点1）
t_range = np.linspace(1e-6, 220*60, 100)  # 从接近0开始，避免除零错误
C_pred = [ade_solution(t) for t in t_range]  # 移除条件判断（关键修改点2）

# 绘图
plt.figure()
plt.scatter(t_exp/60, C_exp, label='Experiment', color='red')
plt.plot(t_range/60, C_pred, label='ADE Prediction', linestyle='--')
plt.xlabel('Time (min)')
plt.ylabel('C/C₀')
plt.legend()
plt.show()