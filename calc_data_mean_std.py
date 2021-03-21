# coding: utf-8
"""
test dataloader

usage: dataloader.py [options]

options:
    --config=<json>                 path of configuration parameter [default: ./config.json]
    --wav_file_path=<path>          path of wav_file [default: F:/Downsample_LJ/*]
    --save_pkl_path=<path>          path of pickle file [default: ./mean_std_pkl/mean_std.pkl]
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
import pickle

from dataloader import Wav2MelF0

if __name__ == "__main__":
    args = docopt(__doc__)
    wav_file_path = args["--wav_file_path"]
    load_wav_to_memory = args["--load_wav_to_memory"]
    save_path = args["--save_pkl_path"]
    with open(args["--config"]) as f:
        data = f.read()
    config = json.loads(data)
    print("start training:")
    data_config = config["data_config"]
    dataloader_config = config["dataloader_config"]
    wav_data = Wav2MelF0(wav_file_path, load_wav_to_memory, **data_config)
    wav_data.val_wav = wav_data.train_wav
    wav, mel, f0 = wav_data.get_all_length_data(0)
    wav_mean = torch.mean(wav)
    wav_var = torch.var(wav)
    mel_mean = torch.mean(mel, dim=1)
    mel_var = torch.var(mel, dim=1)
    f0_mean = torch.mean(f0, dim=1)
    f0_var = torch.var(f0, dim=1)
    for idx in tqdm(range(1, wav_data.__len__())):
        wav, mel, f0 = wav_data.get_all_length_data(idx)
        wav_mean += torch.mean(wav)
        wav_var += torch.var(wav)
        mel_mean += torch.mean(mel, dim=1)
        mel_var += torch.var(mel, dim=1)
        f0_mean += torch.mean(f0, dim=1)
        f0_var += torch.var(f0, dim=1)

    idx += 1
    wav_std = torch.sqrt(wav_var / idx)
    mel_std = torch.sqrt(mel_var / idx)
    f0_std = torch.sqrt(f0_var / idx)
    wav_mean = wav_mean / idx
    mel_mean = mel_mean / idx
    f0_mean = f0_mean / idx

    wav_mean = wav_mean.data.numpy()
    wav_std = wav_std.data.numpy()
    cond_mean = np.append(mel_mean.data.numpy()[0], f0_mean.data.numpy()[0])
    cond_std = np.append(mel_std.data.numpy()[0], f0_std.data.numpy()[0])

    data_mean_std = [cond_mean, cond_std, wav_mean[np.newaxis], wav_std[np.newaxis]]
    with open(save_path, "wb") as f:
        pickle.dump(data_mean_std, f)
