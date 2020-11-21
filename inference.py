# coding: utf-8
"""
inference VDE_wavenet

usage: inference.py [options] <checkpoint> <wav_file_path>

options:
    --config=<json>                 path of configuration parameter [default: ./config.json]
    --save_path=<path>              save output [default: ./inference/]
"""
from docopt import docopt
from tqdm import tqdm
import os
from os.path import join
import numpy as np
import json
import librosa
import librosa.display
import matplotlib
from PIL import Image
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from matplotlib import cm
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

#my modules
from dataloader import Wav2MelF0
from model import Model, Loss
from train import load_checkpoint, prepare_spec_image


use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False

def infer(mel, f0, model_path, data_mean_std_path='ljspeech_mean_std.pkl'):
    device = torch.device("cuda" if use_cuda else "cpu")
    mel, f0 = mel.to(device), f0.to(device)
    with open(data_mean_std_path, 'rb') as f:
        data_mean_std = pickle.load(f)
    model = Model(in_dim=81, out_dim=1, args=None, mean_std=data_mean_std).to(device)
    model, _, _ = load_checkpoint(model_path, model, None, reset_optimizer=True)
    model.eval()

    with torch.no_grad():
        output = model(mel, f0)

    output = output[0].squeeze().cpu().data.numpy()
    mel_output = librosa.feature.melspectrogram(output, **data_config["mel_config"])
    mel_output = np.log(np.abs(mel_output).clip(1e-5, 10)).astype(np.float32)
    mel_output = prepare_spec_image(mel_output)

    return output, mel_output

if __name__ == '__main__':
    args = docopt(__doc__)
    checkpoint_path = args["<checkpoint>"]
    wav_file_path = args["<wav_file_path>"]
    save_dir = args["--save_path"]
    #load config.json
    with open(args["--config"]) as f:
        data = f.read()
    config = json.loads(data)

    print('start inference:')
    global data_config
    data_config = config["data_config"]
    dataloader_config = config["dataloader_config"]
    train_config = config["train_config"]
    dataset = Wav2MelF0(wav_file_path, False, **data_config)

    wav, mel, f0 = dataset.get_all_length_data(0)
    output, mel_output = infer(mel, f0, checkpoint_path)

    #TODO:refactoring
    os.makedirs(save_dir, exist_ok=True)
    check_point_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    audio_name = os.path.splitext(os.path.basename(wav_file_path))[0]
    save_name = 'infer_' + audio_name + '_' + check_point_name + '.wav'
    save_mel_name = 'infer_' + audio_name + '_' + check_point_name + 'mel.png'
    save_path = os.path.join(save_dir, save_name)
    save_mel_path = os.path.join(save_dir, save_mel_name)

    librosa.output.write_wav(save_path, output/0.6, sr=data_config["sampling_rate"])
    Image.fromarray(mel_output).save(save_mel_path)
    librosa.display.waveplot(output, sr=data_config["sampling_rate"])
    plt.savefig(save_path.replace('.wav', '_waveform.png'), format='png')
    plt.close()

    mel = mel[0].cpu().data.numpy()
    mel = prepare_spec_image(mel)
    save_mel_path = os.path.join(save_dir, audio_name+'_mel.png')
    Image.fromarray(mel).save(save_mel_path)
    wav = wav.cpu().data.numpy()
    librosa.display.waveplot(wav, sr=data_config["sampling_rate"])
    save_waveform_path = os.path.join(save_dir, audio_name+'_waveform.png')
    plt.savefig(save_waveform_path, format='png')