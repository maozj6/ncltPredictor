

import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':

    data = np.load('train.npz',allow_pickle=True)
    x = np.array(data['p'],dtype=np.float64)[1:]
    y = np.array(data['h'],dtype=np.float64)[1:]

    # # 计算每一步的速度
    # dx = np.diff(x)
    # dy = np.diff(y)
    # speeds = np.sqrt(dx ** 2 + dy ** 2)
    #
    # # 找到最大速度
    # max_speed = np.max(speeds)
    # 最大速度: 0.086180407172713
    #
    # print("最大速度:", max_speed)
    # # 最大值
    max_value = np.max(x)

    # 最小值
    min_value = np.min(x)
    # 找到 NaN 的位置
    nan_positions = np.where(np.isnan(x))[0]  # [0] 提取具体索引

    # 打印 NaN 的位置
    print("NaN 的位置:", nan_positions)
    print(f"最大值: {max_value}, 最小值: {min_value}")

    # 最大值
    max_value = np.max(y)

    # 最小值
    min_value = np.min(y)
    # 找到 NaN 的位置
    nan_positions = np.where(np.isnan(y))[0]  # [0] 提取具体索引

    # 打印 NaN 的位置
    print("NaN 的位置:", nan_positions)
    print(f"最大值: {max_value}, 最小值: {min_value}")
    print()
    #
    # #