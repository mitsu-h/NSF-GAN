# coding: utf-8
"""
downsample wavfile from 22050 to 16000

usage: dataloader.py [options]

options:
    --config=<json>                 path of configuration parameter [default: ./config.json]
    --wav_file_path=<path>          path of wav_file [default: F:/LJSpeech-1.1/wavs/*]
    --output_dir=<dir>                    output downsample audio file directory [default: F:/Downsample_LJ/]
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
import os
from tqdm import tqdm
from dataloader import print_config_data

if __name__ == "__main__":
    args = docopt(__doc__)
    wav_file_path = args["--wav_file_path"]
    output_dir = args["--output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    with open(args["--config"]) as f:
        data = f.read()
    config = json.loads(data)
    print_config_data(config)

    down_sr = 16000  # 現状固定
    wav_file_path = glob(wav_file_path)
    for wav_path in tqdm(wav_file_path):
        wav, sr = librosa.load(wav_path)
        down_wav = librosa.resample(wav, sr, down_sr)
        filename = os.path.basename(wav_path)
        save_path = os.path.join(output_dir, filename)
        librosa.output.write_wav(save_path, down_wav, down_sr)
