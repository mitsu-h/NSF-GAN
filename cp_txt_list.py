# coding: utf-8
"""
copy the wave file to compare generated

usage: cp_txt_list.py [options]

options:
    --wav_dir=<dir>                 path of wavefile directory [default: F:/norm_lj/]
    --cp_dir=<dir>                  path of wav_file [default: ./自然音声/]
"""

from docopt import docopt
import shutil
import os
from os.path import join
import pickle

if __name__ == "__main__":
    args = docopt(__doc__)
    wav_dir = args["--wav_dir"]
    cp_dir = args["--cp_dir"]

    with open(join(wav_dir, "test_list.txt"), "rb") as f:
        wav_list = pickle.load(f)

    os.makedirs(cp_dir, exist_ok=True)

    for wav_name in wav_list:
        wav_path = join(wav_dir, wav_name)
        shutil.copy(wav_path, cp_dir)
