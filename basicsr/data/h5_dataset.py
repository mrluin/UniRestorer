import h5py
import torch
import numpy as np

from torch.utils.data import Dataset


class H5PairDataset(Dataset):
    def __init__(self, h5_path, lq_key='LQ', gt_key='GT', transform=None):
        """
        h5_path: h5文件路径
        lq_key: 低质量图像所在的组名
        gt_key: 高质量图像所在的组名
        transform: 数据增强（可选）
        """
        self.h5_path = h5_path
        self.lq_key = lq_key
        self.gt_key = gt_key
        self.transform = transform

        # 打开一次，记录数量
        with h5py.File(self.h5_path, 'r') as f:
            self.length = len(f[self.lq_key])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 训练过程中避免每次重新打开，改成lazy open
        with h5py.File(self.h5_path, 'r') as f:
            lq = f[self.lq_key][str(idx)][()]
            gt = f[self.gt_key][str(idx)][()]

        # 转为float32且归一化
        lq = lq.astype(np.float32) / 255.0
        gt = gt.astype(np.float32) / 255.0

        # 转成tensor，注意是 C x H x W 格式
        lq = torch.from_numpy(lq).permute(2, 0, 1)
        gt = torch.from_numpy(gt).permute(2, 0, 1)

        # 做数据增强（比如随机flip、crop等）
        if self.transform:
            lq, gt = self.transform(lq, gt)

        return lq, gt