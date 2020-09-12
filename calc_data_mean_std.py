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

from dataloader import Wav2MelF0

if __name__ == '__main__':
    args = docopt(__doc__)
    wav_file_path = args["--wav_file_path"]
    load_wav_to_memory = args["--load_wav_to_memory"]
    with open(args["--config"]) as f:
        data = f.read()
    config = json.loads(data)
    print('start training:')
    data_config = config["data_config"]
    dataloader_config = config["dataloader_config"]
    wav_data = Wav2MelF0(wav_file_path, load_wav_to_memory, **data_config)

    for idx in tqdm(range(wav_data.__len__())):
        wav, cond = wav_data[idx]
        wav_mean = torch.mean(wav)
        wav_var = torch.var(wav)
