import numpy as np

from torch import Tensor
import torchaudio

class FilterBankFeatureTransform():

    def __init__(self, num_mels, window_length, window_stride):
        super(FilterBankFeatureTransform, self).__init__()
        self.num_mels = num_mels
        self.window_length = window_length
        self.window_stride = window_stride
        self.function = torchaudio.compliance.kaldi.fbank

    def __call__(self, signal):
        return self.function(
            Tensor(signal).unsqueeze(0),
            num_mel_bins=self.num_mels,
            frame_length=self.window_length,
            frame_shift=self.window_stride
        ).transpose(0, 1).numpy()