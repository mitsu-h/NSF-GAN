# coding: utf-8
"""
train VDE_wavenet

usage: train.py [options]

options:
    --config=<json>                 path of configuration parameter [default: ./config.json]
    --wav_file_path=<path>          path of wav_file [default: F:/Downsample_LJ/*]
    --load_wav_to_memory            Do you want to load all wavefile?
"""
from docopt import docopt
from tqdm import tqdm
import os
from os.path import join
import numpy as np
import json
import librosa
import librosa.display
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

#my modules
from dataloader import Wav2MelF0
from model import Model, Loss
import config as prj_conf


use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False

def prepare_spec_image(spectrogram):
    # [0, 1]
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    spectrogram = np.flip(spectrogram, axis=0)  # flip against freq axis
    return np.uint8(cm.magma(spectrogram) * 255)

def prepare_melspec(wav, segment_length=8000, sampling_rate=22050,
                    n_fft=1024, win_length=1024,hop_length=256,n_mels=80, fmin=0.0, fmax=8000.0, power=1.0,
                    f0_frame_period=5.8):
    mel = librosa.feature.melspectrogram(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                         n_mels=n_mels, fmin=fmin, fmax=fmax,power=power)
    mel = np.log(np.abs(mel).clip(1e-5, 10)).astype(np.float32)
    mel = prepare_spec_image(mel)
    return mel.transpose(2, 0, 1)

def save_checkpoint(model, optimizer, total_step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(total_step))
    optimizer_state = optimizer.state_dict()
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "total_step": total_step,
        "epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    total_step = checkpoint["total_step"]
    epoch = checkpoint["epoch"]

    return model, total_step, epoch

def eval_model(step, writer, device, model, eval_data, checkpoint_dir, segment_length=16000, sampling_rate=22050,
               n_fft=512, win_length=512, hop_length=128, n_mels=80, fmin=0.0, fmax=8000, power=1.0, f0_frame_period=5.8):
    target_wav, mel, f0 = eval_data
    mel, f0 = mel.to(device), f0.to(device)

    #prepare model for evaluation
    model_eval = Model(in_dim=n_mels+1, out_dim=1, args=None).to(device)
    model_eval.load_state_dict(model.state_dict())
    model_eval.eval()

    with torch.no_grad():
        output, har_signal, noi_signal = model_eval(mel, f0)
    #save
    output = output[0].cpu().data.numpy()
    mel_output = librosa.feature.melspectrogram(output, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                                n_mels=n_mels, fmin=fmin, fmax=fmax, power=power)
    mel_output = np.log(np.abs(mel_output).clip(1e-5, 10)).astype(np.float32)
    mel_output = prepare_spec_image(mel_output)
    writer.add_image('predicted wav mel spectrogram', mel_output.transpose(2, 0, 1), step)

    mel = mel[0].cpu().data.numpy()
    mel = prepare_spec_image(mel.T)
    writer.add_image('target wav mel spectrogram', mel.transpose(2, 0, 1), step)

    fig = plt.figure()
    librosa.display.waveplot(output, sr=data_config["sampling_rate"])
    writer.add_figure('output', fig, step)
    plt.close()
    output /= np.max(np.abs(output))
    writer.add_audio('predicted audio signal', output, step, sample_rate=data_config["sampling_rate"])
    path = join(checkpoint_dir, 'predicted_signal_checkpoint{}.wav'.format(step))
    librosa.output.write_wav(path, output, sr=data_config["sampling_rate"])

    # save natural audio
    target_wav = target_wav.cpu().data.numpy()
    fig = plt.figure()
    librosa.display.waveplot(target_wav, sr=data_config["sampling_rate"])
    writer.add_figure('target', fig, step)
    plt.close()
    target_wav /= np.max(np.abs(target_wav))
    writer.add_audio('natural audio', target_wav, step, sample_rate=data_config["sampling_rate"])

    # save other output
    noi_signal = noi_signal[0].cpu().data.numpy()
    # ax = save_wavform_fig(noi_signal)
    fig = plt.figure()
    librosa.display.waveplot(noi_signal, sr=data_config["sampling_rate"])
    writer.add_figure('noise signal output', fig, step)
    plt.close()

    writer.add_image('noise signal mel spectrogram', prepare_melspec(noi_signal, **data_config), step)

    har_signal = har_signal[0].cpu().data.numpy()
    fig = plt.figure()
    librosa.display.waveplot(har_signal, sr=data_config["sampling_rate"])
    writer.add_figure('harmonic signal output', fig, step)
    plt.close()

    writer.add_image('harmonic signal mel spectrogram', prepare_melspec(har_signal, **data_config), step)

def train(dataset, train_loader, checkpoint_dir, log_event_path, nepochs,
          learning_rate, eval_per_step, checkpoint_path,seed, data_mean_std_path):
    torch.manual_seed(seed)
    criterion = Loss(None)
    device = torch.device("cuda" if use_cuda else "cpu")

    #Model
    with open(data_mean_std_path, 'rb') as f:
        data_mean_std = pickle.load(f)
    model = Model(in_dim=81, out_dim=1, args=None, mean_std=data_mean_std).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_event_path)

    #train
    epoch = 1
    total_step = 0

    os.makedirs(checkpoint_dir, exist_ok=True)

    if checkpoint_path != "":
        model, total_step, epoch = load_checkpoint(checkpoint_path, model, optimizer)
    while epoch <= nepochs:
        running_loss = 0
        print("{}epoch:".format(epoch))
        for step, (wav, mel, f0) in tqdm(enumerate(train_loader)):
            model.train()
            optimizer.zero_grad()

            wav, mel, f0 = wav.to(device), mel.to(device), f0.to(device)
            outputs = model(mel, f0)

            loss = criterion.compute([outputs[0][:,:wav.size(-1)], outputs[1]], wav)
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss',float(loss.item()), total_step)
            total_step += 1
            running_loss += loss.item()

            if total_step % eval_per_step == 0:
                idx = np.random.randint(0, train_loader.__len__())
                eval_model(total_step, writer, device, model, dataset.get_all_length_data(idx), checkpoint_dir, **data_config)
                save_checkpoint(model, optimizer, total_step, checkpoint_dir, epoch)

        averaged_loss = running_loss / (len(train_loader))
        writer.add_scalar("loss (per epoch)", averaged_loss, epoch)
        print("Loss: {}".format(running_loss / (len(train_loader))))
        epoch += 1

if __name__ == '__main__':
    args = docopt(__doc__)
    wav_file_path = args["--wav_file_path"]
    load_wav_to_memory = args["--load_wav_to_memory"]
    #load config.json
    with open(args["--config"]) as f:
        data = f.read()
    config = json.loads(data)

    print('start training:')
    global data_config
    data_config = config["data_config"]
    dataloader_config = config["dataloader_config"]
    train_config = config["train_config"]
    global wavenet_config
    wavenet_config = config["network_config"]["wavenet_config"]
    dataset = Wav2MelF0(wav_file_path, load_wav_to_memory, **data_config)
    train_loader = DataLoader(dataset, **dataloader_config)
    train(dataset, train_loader, **train_config)


