# coding: utf-8
"""
Divide dataset based on validation and testuation audio length.
This script output divide wavefile list by txtfile.

usage: div_dataset.py [options]

options:
    --config=<json>                 path of configuration parameter [default: ./config.json]
    --wav_file_path=<path>          path of wav_file [default: F:/Downsample_LJ/*]
    --val_wav_hours=<float>         validation audio length [default: 1.0]
    --test_wav_hours=<float>        testuation audio length [default: 1.0]
    --save_txtfile_dir=<dir>        path of txtfile directory [default: F:/Downsample_LJ/]
"""
from docopt import docopt
from glob import glob
import json
import librosa
from os.path import basename, join
import random
import pickle


def devide_wav_list(
    wav_list, wav_hours, sampling_rate, save_txtfile_dir, save_txtfile_name
):
    div_data_hours = 0
    div_list = []
    while True:
        wav, sr = librosa.load(wav_list[0], sr=sampling_rate)
        div_list.append(basename(wav_list.pop(0)))
        div_data_hours += len(wav) / (sr * 3600)
        if div_data_hours > wav_hours:
            div_data_info = f"{save_txtfile_name.replace('_list.txt', '')}: {len(div_list)} utterances, {div_data_hours} hours"
            with open(join(save_txtfile_dir, save_txtfile_name), "wb") as f:
                pickle.dump(div_list, f)
            break
    return div_data_info, div_data_hours


if __name__ == "__main__":
    args = docopt(__doc__)
    wav_file_path = args["--wav_file_path"]
    val_wav_hours = float(args["--val_wav_hours"])
    test_wav_hours = float(args["--test_wav_hours"])
    save_txtfile_dir = args["--save_txtfile_dir"]
    # load config.json
    with open(args["--config"]) as f:
        data = f.read()
    config = json.loads(data)

    wav_list = glob(wav_file_path)
    random.shuffle(wav_list)

    # make validation list
    val_info, val_hours = devide_wav_list(
        wav_list,
        val_wav_hours,
        config["data_config"]["sampling_rate"],
        save_txtfile_dir,
        "val_list.txt",
    )

    # make test list
    test_info, test_hours = devide_wav_list(
        wav_list,
        test_wav_hours,
        config["data_config"]["sampling_rate"],
        save_txtfile_dir,
        "test_list.txt",
    )

    # make train_list
    train_list = [basename(wav_path) for wav_path in wav_list]
    train_info = (
        f"train: {len(train_list)} utterances, {24 - val_hours - test_hours} hours"
    )
    with open(join(save_txtfile_dir, "train_list.txt"), "wb") as f:
        pickle.dump(train_list, f)

    # write divide dataset information
    divide_info = "\n".join([train_info, val_info, test_info])
    print(divide_info)
    with open(join(save_txtfile_dir, "divide_wav_info.txt"), "w") as f:
        f.write(divide_info)
