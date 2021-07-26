import numpy as np

import torch
from torch.utils.data import DataLoader, Sampler

def _collate_fn(batch, pad_token=0):
    # print("batch[list 형식]: ", np.array(batch).shape) # (16, 2) 16=batch, 2=Tensor + transcript
    # print("batch[list 형식]: ", batch[1][0].size()) # (161, Freame)

    def seq_length_(p): # todo 용도
        return len(p[0])
    def target_length_(p): # todo 용도
        return len(p[1])

    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    seq_lengths = [len(s[0]) for s in batch]
    # print("seq_lengths: ", seq_lengths)
    target_lengths = [len(s[1]) for s in batch]
    # print("target_lengths: ", target_lengths)

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(pad_token)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)
    target_lengths = torch.IntTensor(target_lengths)
    # print("targets1: ", targets.size())

    return seqs, targets, seq_lengths, target_lengths

class AudioDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
