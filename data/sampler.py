import numpy as np
from torch.utils.data.sampler import Sampler


class BucketingSampler(Sampler):

    def __init__(self, data_source, batch_size=1):
        """
        비슷한 크기의 samples과 함께 순서대로 배치
        data_source: Dataset
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        # print("data_source: ", len(data_source)) = 59662
        ids = list(range(0, len(data_source))) # idx 만듬
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
        # batch_size 만큼 쪼개짐
        # e.g) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] 와 batch_size=3
        # -> [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19]]
        # print("bins: ", len(self.bins)) batch:16 ==> bin: 3729

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)