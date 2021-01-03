# coding: utf-8
"""
train VDE_wavenet

usage: train.py [options]

options:
    --config=<json>                 path of configuration parameter [default: ./config.json]
    --wav_file_path=<path>          path of wav_file [default: F:/norm_lj/]
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

# my modules
from dataloader import Wav2MelF0
from model import Model, Loss
from model import MelGANMultiScaleDiscriminator as Discriminator
import config as prj_conf


use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False


def prepare_spec_image(spectrogram):
    # [0, 1]
    spectrogram = (spectrogram - np.min(spectrogram)) / (
        np.max(spectrogram) - np.min(spectrogram)
    )
    spectrogram = np.flip(spectrogram, axis=0)  # flip against freq axis
    return np.uint8(cm.magma(spectrogram) * 255)


def prepare_melspec(wav, mel_config):
    mel = librosa.feature.melspectrogram(wav, **mel_config)
    mel = np.log(np.abs(mel).clip(1e-5, 10)).astype(np.float32)
    mel = prepare_spec_image(mel)
    return mel.transpose(2, 0, 1)


def save_checkpoint(
    model,
    optimizer,
    discriminator,
    discriminator_optim,
    total_step,
    checkpoint_dir,
    epoch,
):
    # Generator
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(total_step)
    )
    optimizer_state = optimizer.state_dict()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer_state,
            "total_step": total_step,
            "epoch": epoch,
        },
        checkpoint_path,
    )

    # Discriminator
    disc_path = join(checkpoint_dir, "discriminator_step{:09d}.pth".format(total_step))
    optimizer_state = discriminator_optim.state_dict()
    torch.save(
        {
            "state_dict": discriminator.state_dict(),
            "optimizer": optimizer_state,
            "total_step": total_step,
            "epoch": epoch,
        },
        disc_path,
    )

    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
    return checkpoint


def load_checkpoint(
    path,
    model,
    optimizer,
    discriminator=None,
    discriminator_optim=None,
    reset_optimizer=False,
):
    print("Load checkpoint from: {}".format(path))
    # Generator
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    # Discriminator
    if discriminator is not None:
        disc_checkpoint = _load(path.replace("checkpoint_", "discriminator_"))
        discriminator.load_state_dict(disc_checkpoint["state_dict"])
        if not reset_optimizer:
            optimizer_state = disc_checkpoint["optimizer"]
            if optimizer_state is not None:
                print(
                    "Load optimizer state from {}".format(
                        path.replace("checkpoint_", "discriminator_")
                    )
                )
                discriminator_optim.load_state_dict(disc_checkpoint["optimizer"])

    total_step = checkpoint["total_step"]
    epoch = checkpoint["epoch"]

    return model, discriminator, total_step, epoch


def eval_model(step, writer, device, model, eval_data, checkpoint_dir, mel_config):
    target_wav, mel, f0 = eval_data
    mel, f0 = mel.to(device), f0.to(device)

    # prepare model for evaluation
    model_eval = Model(in_dim=81, out_dim=1, args=None).to(device)
    model_eval.load_state_dict(model.state_dict())
    model_eval.remove_weight_norm()
    model_eval.eval()

    with torch.no_grad():
        output = model_eval(mel, f0)
    # save
    output = output[0].cpu().data.numpy()
    mel_output = librosa.feature.melspectrogram(output, **mel_config)
    mel_output = np.log(np.abs(mel_output).clip(1e-5, 10)).astype(np.float32)
    mel_output = prepare_spec_image(mel_output)
    writer.add_image(
        "predicted wav mel spectrogram", mel_output.transpose(2, 0, 1), step
    )

    mel = mel[0].cpu().data.numpy()
    mel = prepare_spec_image(mel.T)
    writer.add_image("target wav mel spectrogram", mel.transpose(2, 0, 1), step)

    fig = plt.figure()
    librosa.display.waveplot(output, sr=data_config["sampling_rate"])
    writer.add_figure("output", fig, step)
    plt.close()
    # output /= np.max(np.abs(output))
    writer.add_audio(
        "predicted audio signal", output, step, sample_rate=data_config["sampling_rate"]
    )
    path = join(checkpoint_dir, "predicted_signal_checkpoint{}.wav".format(step))
    librosa.output.write_wav(path, output, sr=data_config["sampling_rate"])

    # save natural audio
    target_wav = target_wav.cpu().data.numpy()
    fig = plt.figure()
    librosa.display.waveplot(target_wav, sr=data_config["sampling_rate"])
    writer.add_figure("target", fig, step)
    plt.close()
    # target_wav /= np.max(np.abs(target_wav))
    writer.add_audio(
        "natural audio", target_wav, step, sample_rate=data_config["sampling_rate"]
    )


def train(
    dataset,
    train_loader,
    checkpoint_dir,
    log_event_path,
    nepochs,
    learning_rate,
    eval_per_step,
    generator_step,
    discriminator_step,
    checkpoint_path,
    seed,
    data_mean_std_path,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    criterion = Loss(device)

    # Model
    with open(data_mean_std_path, "rb") as f:
        data_mean_std = pickle.load(f)
    model = Model(in_dim=81, out_dim=1, args=None, mean_std=data_mean_std).to(device)
    discriminator = Discriminator().to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    discriminator_optim = optim.Adam(discriminator.parameters(), lr=learning_rate / 2)

    writer = SummaryWriter(log_event_path)

    # train
    epoch = 1
    total_step = 0

    os.makedirs(checkpoint_dir, exist_ok=True)

    if checkpoint_path != "":
        model, discriminator, total_step, epoch = load_checkpoint(
            checkpoint_path, model, optimizer, discriminator, discriminator_optim
        )
    while epoch <= nepochs:
        running_loss = 0
        print("{}epoch:".format(epoch))
        for step, (wav, mel, f0) in tqdm(enumerate(train_loader)):
            model.train()
            discriminator.train()
            optimizer.zero_grad()
            discriminator_optim.zero_grad()

            wav, mel, f0 = wav.to(device), mel.to(device), f0.to(device)

            # Generator
            if (
                total_step < generator_step
                or total_step > generator_step + discriminator_step
            ):
                outputs = model(mel, f0)

                stft_loss = criterion.stft_loss(outputs[:, : wav.size(-1)], wav)
                if total_step < generator_step:
                    loss = stft_loss
                    adv_loss = None
                else:
                    adv = discriminator(outputs.unsqueeze(1))
                    adv_loss = criterion.adversarial_loss(adv)
                    loss = stft_loss + 0.1 * adv_loss
                loss.backward()
                optimizer.step()
            else:
                loss = None
                stft_loss = None
                adv_loss = None

            # Discriminator
            if total_step > generator_step:
                with torch.no_grad():
                    outputs = model(mel, f0)
                real = discriminator(wav.unsqueeze(1))
                fake = discriminator(outputs.unsqueeze(1).detach())
                real_loss, fake_loss = criterion.discriminator_loss(real, fake)
                dis_loss = real_loss + fake_loss
                dis_loss.backward()
                discriminator_optim.step()
            else:
                dis_loss = None

            if loss is not None:
                writer.add_scalar("loss", float(loss.item()), total_step)
                writer.add_scalar("stft_loss", float(stft_loss.item()), total_step)
            if adv_loss is not None:
                writer.add_scalar("adv_loss", float(adv_loss.item()), total_step)
            if dis_loss is not None:
                writer.add_scalar("dis_loss", float(dis_loss.item()), total_step)
                writer.add_scalar("real_loss", float(real_loss.item()), total_step)
                writer.add_scalar("fake_loss", float(fake_loss.item()), total_step)
            total_step += 1
            # running_loss += loss.item()

            if total_step % eval_per_step == 0:
                idx = np.random.randint(0, len(dataset.val_wav))
                eval_model(
                    total_step,
                    writer,
                    device,
                    model,
                    dataset.get_all_length_data(idx),
                    checkpoint_dir,
                    data_config["mel_config"],
                )
                save_checkpoint(
                    model,
                    optimizer,
                    discriminator,
                    discriminator_optim,
                    total_step,
                    checkpoint_dir,
                    epoch,
                )

        # averaged_loss = running_loss / (len(train_loader))
        # writer.add_scalar("loss (per epoch)", averaged_loss, epoch)
        # print("Loss: {}".format(running_loss / (len(train_loader))))
        epoch += 1


if __name__ == "__main__":
    args = docopt(__doc__)
    wav_file_path = args["--wav_file_path"]
    load_wav_to_memory = args["--load_wav_to_memory"]
    # load config.json
    with open(args["--config"]) as f:
        data = f.read()
    config = json.loads(data)

    print("start training:")
    global data_config
    data_config = config["data_config"]
    dataloader_config = config["dataloader_config"]
    train_config = config["train_config"]
    global wavenet_config
    wavenet_config = config["network_config"]["wavenet_config"]
    dataset = Wav2MelF0(wav_file_path, load_wav_to_memory, **data_config)
    train_loader = DataLoader(dataset, **dataloader_config)
    train(dataset, train_loader, **train_config)
