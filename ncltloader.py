import numpy as np
import torch
class SequenceLoader:
    def __init__(self, npz_path, input_length, label_length):
        """
        初始化加载器

        Args:
            npz_path (str): npz 文件路径
            input_length (int): 输入序列的长度
            label_length (int): 标签序列的长度
        """
        self.data = np.load(npz_path,allow_pickle=True)
        self.input_length = input_length
        self.label_length = label_length

        # 将所有变量转为 numpy 数组
        self.x = self.data['x'][1:]
        self.y = self.data['y'][1:]
        self.z = self.data['z'][1:]
        self.r = self.data['r'][1:]
        self.p = self.data['p'][1:]
        self.h = self.data['h'][1:]

        # 合并数据为一个数组 shape=(N, 6)
        self.sequences = np.stack([self.x, self.y, self.z, self.r, self.p, self.h], axis=-1)

    def __len__(self):
        """返回可用的序列总数"""
        total_length = len(self.sequences)
        return max(0, total_length - self.input_length - self.label_length + 1)

    def __getitem__(self, index):
        """
        获取指定索引的输入序列和标签序列

        Args:
            index (int): 索引值

        Returns:
            tuple: (输入序列, 标签序列)，shape 分别为 (input_length, 6) 和 (label_length, 6)
        """
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        start_idx = index
        end_idx = index + self.input_length + self.label_length

        sequence = self.sequences[start_idx:end_idx]
        input_seq = sequence[:self.input_length]
        label_seq = sequence[self.input_length:]
        inputnumpy = np.array(input_seq[:,1:3],dtype=float)
        labelnumpy = np.array(label_seq[:,1:3],dtype=float)

        inputseq =  torch.tensor(inputnumpy, dtype=torch.float32)
        labelseq =  torch.tensor(labelnumpy, dtype=torch.float32)

        return inputseq, labelseq

# 使用示例
if __name__ == "__main__":
    loader = SequenceLoader("train.npz", input_length=10, label_length=5)

    print(f"Total sequences: {len(loader)}")

    for i in range(len(loader)):
        input_seq, label_seq = loader[i]
        print(f"Input (shape={input_seq.shape}):\n", input_seq)
        print(f"Label (shape={label_seq.shape}):\n", label_seq)
        break
