import librosa
import numpy as np
import os

import torch
import torchaudio
from torch.utils.data import Dataset

from data.filterbank import FilterBankFeatureTransform
from data.augment import spec_augment

def load_audio(audio_path, sample_rate):
    assert audio_path.endswith('wav'), "only wav files"
    signal, sr = librosa.load(audio_path, sr=sample_rate)

    return signal


class SpectrogramDataset(Dataset):

    def __init__(self, audio_conf, dataset_path, data_list, char2index, sos_id, eos_id, normalize=False, mode='train'):
        """
        Dataset 은 wav_name, transcripts, speaker_id 가 dictionary 로 담겨져있는 list으로부터 data 를 load
        :param audio_conf: Sample rate, window, window size나 length, stride 설정
        :param data_list: dictionary . key: 'wav', 'text', 'speaker_id'
        :param char2index: character 에서 index 로 mapping 된 Dictionary
        :param normalize: Normalized by instance-wise standardazation
        """
        super(SpectrogramDataset, self).__init__()
        self.audio_conf = audio_conf # dict{sample rate, window_size, window_stride}
        self.data_list = data_list # [{"wav": , "text": , "speaker_id": "}]
        self.size = len(self.data_list) # 59662
        self.char2index = char2index
        self.sos_id = sos_id # 2001
        self.eos_id = eos_id # 2002
        self.PAD = 0
        self.normalize = normalize # Train: True
        self.dataset_path = dataset_path # data/wavs_train
        self.transforms = FilterBankFeatureTransform(
            audio_conf["num_mel"], audio_conf["window_length"], audio_conf["window_stride"]
        )
        self.mode = mode


    def __getitem__(self, index):
        wav_name = self.data_list[index]['wav']
        # print("wav: " , wav_name): 41_0607_213_1_08139_05.wav
        audio_path = os.path.join(self.dataset_path, wav_name)
        # print("audio_path: ", audio_path): data/wavs_train/41_0607_213_1_08139_05.wav
        transcript = self.data_list[index]['text']
        # print("text: ", transcript): 예약 받나요?

        spect = self.parse_audio(audio_path)
        # print("spect: ", spect.size()): Tensor[161, Frame] 161: 1+ n_fft/2 | Frame: length / stride
        transcript = self.parse_transcript(transcript)
        # print("text: ", transcript): [2001, 22, 3, ..., 2002] 각 단어의 인덱스 2001(sos)

        return spect, transcript


    def parse_audio(self, audio_path):
        signal = load_audio(audio_path, sample_rate=self.audio_conf['sample_rate'])

        feature = self.transforms(signal)

        # normalize
        feature -= feature.mean()
        feature /= np.std(feature)

        feature = torch.FloatTensor(feature).transpose(0, 1)

        # todo basic 우선 먼저 확인
        # if self.mode == 'train':
        #    feature = spec_augment(feature)

        return feature


    def parse_transcript(self, transcript):
        # print(list(transcript))
        # ['아', '기', '랑', ' ', '같', '이', ' ', '갈', '건', '데', '요', ',', ' ', '아', '기', '가', ' ', '먹', '을', ' ', '수', ' ', '있', '는', '것', '도', ' ', '있', '나', '요', '?']
        # ['매', '장', ' ', '전', '용', ' ', '주', '차', '장', '이', ' ', '있', '나', '요', '?']
        # ['카', '드', ' ', '할', '인', '은', ' ', '신', '용', '카', '드', '만', ' ', '되', '나', '요', '?']
        # ['미', '리', ' ', '예', '약', '하', '려', '고', ' ', '하', '는', '데', '요', '.']

        transcript = list(filter(None, [self.char2index.get(x) for x in list(transcript)]))
        # filter(조건, 순횐 가능한 데이터): char2index 의 key 에 없는 것(None) 다 삭제 해버림
        # print("transcript: ", transcript):[49, 153, 4, 85, 63, 24, 129, 5, 4, 47, 601, 64, 4, 137, 55, 126]

        transcript = [self.sos_id] + transcript + [self.eos_id]
        # [2001, 49, 153, 4, 85, 63, 24, 129, 5, 4, 47, 601, 64, 4, 137, 55, 126, 2002]

        return transcript


    def __len__(self):
        return self.size # 59662





