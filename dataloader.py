# coding: utf-8
"""
test dataloader

usage: dataloader.py [options]

options:
    --config=<json>                 path of configuration parameter [default: ./config.json]
    --wav_file_path=<path>          path of wav_file [default: F:/LJSpeech-1.1/wavs/*]
    --load_wav_to_memory            Do you want to load all wavefile?
"""
from docopt import docopt
import torch
import torch.utils.data
import sys
import numpy as np
import json
from glob import glob
import librosa
import pyworld as pw
from torch.utils.data import DataLoader
from tqdm import tqdm

#TODO: this function move other file
def print_config_data(config):
    for conf in config:
        print('%s:' % conf)
        conf = config[conf]
        for c in conf:
            print('\t %s : %s' % (c, conf[c]))

class Wav2MelF0(torch.utils.data.Dataset):
    def __init__(self, wav_file_path, load_wav_to_memory=False, segment_length=16000, sampling_rate=22050, n_fft=1024, win_length=1024,hop_length=256,
                 n_mels=80, fmin=0.0, fmax=8000.0, power=1.0, f0_frame_period=5.8):
        self.isMemory = load_wav_to_memory
        wav_list = glob(wav_file_path)
        if load_wav_to_memory:
            self.wav = []
            for wave_path in tqdm(wav_list):
                wav, sr = librosa.load(wave_path)
                self.wav.append(wav)
        else:
            self.wav = wav_list

        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.power = power
        self.f0_frame_period = f0_frame_period

    def __getitem__(self, idx):
        wav = self.wav[idx] if self.isMemory else librosa.load(self.wav[idx])[0]
        wav_length = len(wav)
        if wav_length >= self.segment_length:
            max_start = wav_length - self.segment_length
            start = np.random.randint(0, max_start)
            wav = wav[start:start+self.segment_length]
        else:
            wav = np.pad(wav, [0, self.segment_length - wav_length], mode='constant')

        #make mel-spectrogram
        mel = librosa.feature.melspectrogram(wav, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
                                             n_mels=self.n_mels,fmin=self.fmin,fmax=self.fmax,power=self.power)
        mel = np.log(np.abs(mel).clip(1e-5,10)).astype(np.float32)
        #make fundamental frequency
        wav = wav.astype(np.float)
        #wav = np.pad(wav, [0, mel.shape[1] * self.hop_length - self.segment_length], mode='constant').astype(np.float)
        _f0, t = pw.dio(wav, self.sampling_rate, frame_period=self.f0_frame_period)
        f0 = pw.stonemask(wav, _f0, t, self.sampling_rate)
        cond = np.append(mel, [f0.astype(np.float32)], axis=0)
        wav = torch.from_numpy(wav.astype(np.float32))
        cond = torch.from_numpy(cond.T)

        return (wav, cond)

    def __len__(self):
        return len(self.wav)

    def get_all_length_data(self, idx):
        wav = self.wav[idx] if self.isMemory else librosa.load(self.wav[idx])[0]
        # make mel-spectrogram
        mel = librosa.feature.melspectrogram(wav, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
                                             n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, power=self.power)
        mel = np.log(np.abs(mel).clip(1e-5, 10)).astype(np.float32)
        # make fundamental frequency
        wav = wav.astype(np.float)
        _f0, t = pw.dio(wav, self.sampling_rate, frame_period=self.f0_frame_period)
        f0 = pw.stonemask(wav, _f0, t, self.sampling_rate)[:mel.shape[1]*2]
        cond = np.concatenate([mel.T, f0.astype(np.float32)])
        wav = torch.from_numpy(wav.astype(np.float32))
        cond = torch.from_numpy(cond).unsqueeze(0)

        return (wav, cond)

if __name__ == '__main__':
    args = docopt(__doc__)
    wav_file_path = args["--wav_file_path"]
    load_wav_to_memory = args["--load_wav_to_memory"]
    with open(args["--config"]) as f:
        data = f.read()
    config = json.loads(data)
    print_config_data(config)
    print('start training:')
    data_config = config["data_config"]
    dataloader_config = config["dataloader_config"]
    wav_data = Wav2MelF0(wav_file_path, load_wav_to_memory, **data_config)

    train_loader = DataLoader(wav_data, **dataloader_config)

    for i, batch in tqdm(enumerate(train_loader)):
        wav, cond = batch