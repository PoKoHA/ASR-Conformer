import argparse
import array
import math
import numpy as np
import random
import wave
import json
import os


# https://engineering.linecorp.com/ko/blog/voice-waveform-arbitrary-signal-to-noise-ratio-python/

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='../data/train.json')
    parser.add_argument('--clean-file-path', type=str, required=True) # 음성 파일 clovacall
    parser.add_argument('--noise-file-path', type=str, required=True) # noise file
    parser.add_argument('--output-noisy-path', type=str, default='', required=True)
    # 임의의 SN비로 적용된 음성파일 절대경로
    parser.add_argument('--snr', type=float, default='', required=True)
    # SN 비율
    args = parser.parse_args()
    return args

def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes()) # 최대 n개의 오디오 프레임을 읽어들여 bytes 객체 반환
    # getnfreames: 오디오 프레임 수를 반환 ==> wav 파일의 모든 진폭값 취득
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude

# RMS 구하기(진폭값 MSE)
def cal_rms(amp):
    """
    주의할점: 내가 가지고 있는 Noisy File은 5분 / clovacall: 3초에서 최대14초 정도 즉
    clovacall 길이에 맞게 자르고 계산해야함
    """
    return np.sqrt(np.mean(np.square(amp), axis=-1)) # 단순히 RMS식

# SNR 식 이용해 파형 합성
def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20 # a = noise
    noise_rms = clean_rms / (10**a)
    return noise_rms

if __name__ == '__main__':
    args = get_args()

    snr = args.snr

    with open(args.train_file, 'r', encoding='utf-8') as f: # train_file: data/train.json
        trainData_list = json.load(f)

    for index in range(len(trainData_list)):
        wav_name = trainData_list[index]['wav']
        print("wav_name: ", wav_name)
        audio_path = os.path.join(args.clean_file_path, wav_name)
        clean_wav = wave.open(audio_path, "r")

        noise_list = os.listdir(args.noise_file_path)
        noise_wav_name = random.choice(noise_list)
        print("noise_wav_name: ", noise_wav_name)
        noise_path = os.path.join(args.noise_file_path, noise_wav_name)
        noise_wav = wave.open(noise_path, "r")

        clean_amp = cal_amp(clean_wav)
        noise_amp = cal_amp(noise_wav)

        # clean file rms구함
        clean_rms = cal_rms(clean_amp)

        # 임의로 자른 start point 에서 clean_file length 까지
        if len(noise_amp) > len(clean_amp):
            # 자를 위치를 랜덤으로 정함
            start = random.randint(0, len(noise_amp) - len(clean_amp))
            noise_amp = noise_amp[start: start + len(clean_amp)]
            noise_rms = cal_rms(noise_amp)

        else:
            noise_amp = noise_amp
            noise_rms = cal_rms(noise_amp)

        # 그리고 rms 구함
        adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
        adjusted_noise_amp = noise_amp * (adjusted_noise_rms / noise_rms)  # 조정된 noise
        mixed_amp = (clean_amp + adjusted_noise_amp)  # 합성

        # 최대 int16 범위를 넘으면 안되므로
        if (mixed_amp.max(axis=0) > 32767):
            mixed_amp = mixed_amp * (32767 / mixed_amp.max(axis=0))
            clean_amp = clean_amp * (32767 / mixed_amp.max(axis=0))
            adjusted_noise_amp = adjusted_noise_amp * (32767 / mixed_amp.max(axis=0))

        output_path = os.path.join(args.output_noisy_path, wav_name)
        noisy_wave = wave.Wave_write(output_path)  # 자징 걍러
        noisy_wave.setparams(clean_wav.getparams())
        noisy_wave.writeframes(array.array('h', mixed_amp.astype(np.int16)).tostring())
        noisy_wave.close()


















