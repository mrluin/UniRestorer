import math
import torch
import random
import numpy as np
from torch.utils.data.sampler import Sampler


class EnlargedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(
            len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class SameSourceBatchSampler(Sampler):
    def __init__(self, datasets, ratios, batch_size, shuffle=True, seed=0, sampling_strategy='uniform'):
        self.datasets = datasets
        self.ratios = ratios
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.strategy = sampling_strategy

        self.offsets = self._get_offsets()
        self.sub_indices = self._prepare_indices()

        # 用于 uniform 采样的概率
        self.sample_probs = self._get_sampling_probs()

        self.total_batches = self._estimate_total_batches()

    def _get_offsets(self):
        offsets = []
        acc = 0
        for d in self.datasets:
            offsets.append(acc)
            acc += len(d)
        return offsets

    def _prepare_indices(self):
        sub_indices = []
        for dataset, ratio in zip(self.datasets, self.ratios):
            count = int(len(dataset) * ratio)
            sub_indices.append(list(range(count)))
        return sub_indices

    def _get_sampling_probs(self):
        if self.strategy == 'uniform':
            return [1.0 / len(self.datasets)] * len(self.datasets)
        elif self.strategy == 'proportional':
            lens = [int(len(d) * r) for d, r in zip(self.datasets, self.ratios)]
            total = sum(lens)
            return [l / total for l in lens]
        else:
            raise ValueError("sampling_strategy must be 'uniform' or 'proportional'")

    def _estimate_total_batches(self):
        per_dataset = [int(len(d) * r) for d, r in zip(self.datasets, self.ratios)]
        return sum([math.ceil(c / self.batch_size) for c in per_dataset])

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_batches = []

        prepared = []
        for i, (dataset, ratio, offset) in enumerate(zip(self.datasets, self.ratios, self.offsets)):
            total_samples = int(len(dataset) * ratio)
            indices = torch.randperm(total_samples, generator=g).tolist()
            indices = [v % len(dataset) + offset for v in indices]
            prepared.append(indices)

        # 为每个数据集生成 batch 列表
        per_dataset_batches = []
        for indices in prepared:
            batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
            per_dataset_batches.append(batches)

        # 确保每轮 batch 数量一致（可以 clip 或循环补齐）
        min_batches = min(len(b) for b in per_dataset_batches)
        max_batches = max(len(b) for b in per_dataset_batches)

        final_batches = []

        # 用采样概率进行选择
        dataset_indices = list(range(len(self.datasets)))
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch + 123)

        # 尽可能公平地 sample 各个子集的 batch
        while any(per_dataset_batches):
            chosen = random.choices(dataset_indices, weights=self.sample_probs, k=1)[0]
            if per_dataset_batches[chosen]:
                final_batches.append(per_dataset_batches[chosen].pop(0))

        return iter(final_batches)

    def __len__(self):
        return self.total_batches
