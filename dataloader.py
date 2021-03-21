# coding: utf-8
"""
test dataloader

usage: dataloader.py [options]

options:
    --config=<json>                 path of configuration parameter [default: ./config.json]
    --wav_file_path=<dir>          path of wav_file [default: F:/LJSpeech-1.1/wavs/]
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
from os.path import join
import pyworld as pw
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm

# TODO: this function move other file
def print_config_data(config):
    for conf in config:
        print("%s:" % conf)
        conf = config[conf]
        for c in conf:
            print("\t %s : %s" % (c, conf[c]))


class Wav2MelF0(torch.utils.data.Dataset):
    def __init__(
        self,
        wav_file_path,
        load_wav_to_memory,
        segment_length,
        sampling_rate,
        f0_frame_period,
        mel_config,
    ):
        self.isMemory = load_wav_to_memory
        self.wav_file_dir = wav_file_path
        with open(join(wav_file_path, "train_list.txt"), "rb") as f:
            self.train_wav = pickle.load(f)
        with open(join(wav_file_path, "val_list.txt"), "rb") as f:
            self.val_wav = pickle.load(f)
        with open(join(wav_file_path, "test_list.txt"), "rb") as f:
            self.test_wav = pickle.load(f)

        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.f0_frame_period = f0_frame_period
        self.mel_config = mel_config

    def make_mel_f0(self, wav):
        # make mel-spectrogram
        mel = librosa.feature.melspectrogram(wav, self.sampling_rate, **self.mel_config)
        mel = np.log(np.abs(mel).clip(1e-5, 10)).astype(np.float32)
        # make fundamental frequency
        wav = wav.astype(np.float)
        _f0, t = pw.dio(wav, self.sampling_rate, frame_period=self.f0_frame_period)
        f0 = pw.stonemask(wav, _f0, t, self.sampling_rate)
        wav = torch.from_numpy(wav.astype(np.float32))
        mel = torch.from_numpy(mel).T
        f0 = torch.from_numpy(f0.astype(np.float32)).unsqueeze(-1)
        return wav, mel, f0

    def __getitem__(self, idx):
        wav = librosa.load(
            join(self.wav_file_dir, self.train_wav[idx]), sr=self.sampling_rate
        )[0]
        wav_length = len(wav)
        if wav_length >= self.segment_length:
            max_start = wav_length - self.segment_length
            start = np.random.randint(0, max_start)
            wav = wav[start : start + self.segment_length]
        else:
            wav = np.pad(wav, [0, self.segment_length - wav_length], mode="constant")

        wav, mel, f0 = self.make_mel_f0(wav)

        return (wav, mel, f0)

    def __len__(self):
        return len(self.train_wav)

    def get_all_length_data(self, idx, is_test=False):
        if not is_test:
            wav = librosa.load(
                join(self.wav_file_dir, self.val_wav[idx]), sr=self.sampling_rate
            )[0]

        else:
            wav = librosa.load(
                join(self.wav_file_dir, self.test_wav[idx]), sr=self.sampling_rate
            )[0]

        wav, mel, f0 = self.make_mel_f0(wav)
        mel = mel.unsqueeze(0)
        f0 = f0.unsqueeze(0)

        return (wav, mel, f0)


if __name__ == "__main__":
    args = docopt(__doc__)
    wav_file_path = args["--wav_file_path"]
    load_wav_to_memory = args["--load_wav_to_memory"]
    with open(args["--config"]) as f:
        data = f.read()
    config = json.loads(data)
    print_config_data(config)
    print("start training:")
    data_config = config["data_config"]
    dataloader_config = config["dataloader_config"]
    wav_data = Wav2MelF0(wav_file_path, load_wav_to_memory, **data_config)

    train_loader = DataLoader(wav_data, **dataloader_config)

    for i, batch in tqdm(enumerate(train_loader)):
        wav, mel, f0 = batch
