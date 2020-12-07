# coding: utf-8
"""
div_dataset

usage: div_dataset.py [options]

options:
    --config=<json>                 path of configuration parameter [default: ./config.json]
    --wav_file_path=<path>          path of wav_file [default: F:/Downsample_LJ/*]
    --save_txtfile_dir=<dir>        path of txtfile directory [default: F:/Downsample_LJ/]
"""
from docopt import docopt
from glob import glob
import json
import librosa
from os.path import basename
import random
import pickle

if __name__ == '__main__':
    args = docopt(__doc__)
    wav_file_path = args["--wav_file_path"]
    save_txtfile_dir = args["--save_txtfile_dir"]
    # load config.json
    with open(args["--config"]) as f:
        data = f.read()
    config = json.loads(data)
    
    wav_list = glob(wav_file_path)
    random.shuffle(wav_list)
    val_list = []
    test_list = []
    val_hours = 0
    test_hours = 0

    #make validation list
    while True:
        wav, sr = librosa.load(wav_list[0], sr=config["data_config"]["sampling_rate"])
        val_list.append(basename(wav_list.pop(0)))
        val_hours += len(wav) / (sr * 3600)
        if val_hours > 1:
            print(f'validation: {len(val_list)} utterances, {val_hours} hours')
            with open(save_txtfile_dir+'val_list.txt', 'wb') as f:
                pickle.dump(val_list, f)
            break

    #make test list
    while True:
        wav, sr = librosa.load(wav_list[0], sr=config["data_config"]["sampling_rate"])
        test_list.append(basename(wav_list.pop(0)))
        test_hours += len(wav) / (sr * 3600)
        if test_hours > 1:
            print(f'test: {len(test_list)} utterances, {test_hours} hours')
            with open(save_txtfile_dir+'test_list.txt', 'wb') as f:
                pickle.dump(test_list, f)
            break

    #make train_list
    train_list = [basename(wav_path) for wav_path in wav_list]
    print(f'train: {len(train_list)} utterances, {24 - val_hours - test_hours} hours')
    with open(save_txtfile_dir + 'train_list.txt', 'wb') as f:
        pickle.dump(train_list, f)