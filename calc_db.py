# coding: utf-8
"""
calcurate mean dB in the directory

usage: calc_db.py [options] <wav_dir>

options:
    --sr=<int>      sampling rate [default: 16000]
"""

from docopt import docopt
import numpy as np
import librosa
from glob import glob
from os.path import join
from tqdm import tqdm


def calc_rms(wav):
    return np.sqrt(sum(wav ** 2) / len(wav))


def calc_mean_db(wav_list, sr):
    db_mean = 0
    for i, wav_name in enumerate(tqdm(wav_list)):
        wav, _ = librosa.load(wav_name, sr=sr)
        db_mean = (db_mean * i + calc_rms(wav)) / (i + 1)
    return 20 * np.log10(db_mean)


# natural db
if __name__ == "__main__":
    args = docopt(__doc__)
    wav_dir = args["<wav_dir>"]
    sr = int(args["--sr"])
    wav_list = glob(join(wav_dir, "*.wav"))

    wav_mean_db = calc_mean_db(wav_list, sr)

    print(f"mean dB in the {wav_dir}:{wav_mean_db}[dB]")
