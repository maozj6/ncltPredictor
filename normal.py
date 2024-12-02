

import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':

    data = np.load('train.npz',allow_pickle=True)
    x1 = np.array(data['y'],dtype=np.float64)[1:]
    y1 = np.array(data['z'],dtype=np.float64)[1:]

    data2 = np.load('val.npz',allow_pickle=True)
    x2 = np.array(data2['y'],dtype=np.float64)[1:]
    y2 = np.array(data2['z'],dtype=np.float64)[1:]

    # 创建散点图
    plt.figure(figsize=(12, 10))
    plt.scatter(x1, y1, s=0.1, alpha=0.5, c='blue', label='Dataset 1')  # 数据集1 - 蓝点
    plt.scatter(x2, y2, s=0.1, alpha=0.5, c='red', label='Dataset 2')  # 数据集2 - 红点

    # 添加标题和标签
    plt.title('Scatter Plot of Two Large Datasets')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 添加图例和网格
    plt.legend()
    plt.grid(True)

    # 显示图像
    plt.show()
    #
    # 代码关键点：
    # # 绘制散点图
    # plt.figure(figsize=(10, 8))
    # plt.scatter(x, y, s=0.1, alpha=0.5, c='blue')  # s: 点大小, alpha: 透明度
    # plt.title('Scatter Plot of 600,000 Points')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.grid(True)
    # plt.show()
    # # 最大值
    # max_value = np.max(x)
    #
    # # 最小值
    # min_value = np.min(x)
    # # 找到 NaN 的位置
    # nan_positions = np.where(np.isnan(x))[0]  # [0] 提取具体索引
    #
    # # 打印 NaN 的位置
    # print("NaN 的位置:", nan_positions)
    # print(f"最大值: {max_value}, 最小值: {min_value}")
    #
    # # 最大值
    # max_value = np.max(y)
    #
    # # 最小值
    # min_value = np.min(y)
    # # 找到 NaN 的位置
    # nan_positions = np.where(np.isnan(y))[0]  # [0] 提取具体索引
    #
    # # 打印 NaN 的位置
    # print("NaN 的位置:", nan_positions)
    # print(f"最大值: {max_value}, 最小值: {min_value}")
    # print()
    #
    # #