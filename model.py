#!/usr/bin/env python
"""
model.py for harmonic-plus-noise NSF with trainable sinc filter

version: 9

"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import config as prj_conf
import torch.fft

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


##############
# Building blocks (torch.nn modules + dimension operation)

#
# For blstm
class BLSTMLayer(torch_nn.Module):
    """Wrapper over dilated conv1D
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)

    Recurrency is conducted along "length"
    """

    def __init__(self, input_dim, output_dim):
        super(BLSTMLayer, self).__init__()
        if output_dim % 2 != 0:
            print("Output_dim of BLSTMLayer is {:d}".format(output_dim))
            print("BLSTMLayer expects a layer size of even number")
            sys.exit(1)
        # bi-directional LSTM
        self.l_blstm = torch_nn.LSTM(input_dim, output_dim // 2, bidirectional=True)
        # cannot apply weight_norm. so, add under line
        # https://github.com/pytorch/pytorch/issues/39311
        """
        name_pre = "weight"
        name = name_pre + "_hh_l0"
        torch_nn.utils.weight_norm(self.l_blstm, name)
        name = name_pre + "_ih_l0"
        torch_nn.utils.weight_norm(self.l_blstm, name)
        """

    def forward(self, x):
        # permute to (length, batchsize=1, dim)
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))

        # permute it backt to (batchsize=1, length, dim)
        return blstm_data.permute(1, 0, 2)


#
# 1D dilated convolution that keep the input/output length
class Conv1dKeepLength(torch_nn.Module):
    """Wrapper for causal convolution
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    https://github.com/pytorch/pytorch/issues/1333
    Note: Tanh is optional
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        dilation_s,
        kernel_s,
        causal=False,
        stride=1,
        groups=1,
        bias=False,
        tanh=True,
        pad_mode="constant",
    ):
        super(Conv1dKeepLength, self).__init__()

        self.pad_mode = pad_mode

        self.causal = causal
        # input & output length will be the same
        if self.causal:
            # left pad to make the convolution causal
            self.pad_le = dilation_s * (kernel_s - 1)
            self.pad_ri = 0
        else:
            # pad on both sizes
            self.pad_le = dilation_s * (kernel_s - 1) // 2
            self.pad_ri = dilation_s * (kernel_s - 1) - self.pad_le

        self.conv = torch_nn.Conv1d(
            input_dim,
            output_dim,
            kernel_s,
            stride=stride,
            padding=0,
            dilation=dilation_s,
            groups=groups,
            bias=bias,
        )

        if tanh:
            self.l_ac = torch_nn.Tanh()
        else:
            self.l_ac = torch_nn.Identity()

    def forward(self, data):
        # permute to (batchsize=1, dim, length)
        # add one dimension (batchsize=1, dim, ADDED_DIM, length)
        # pad to ADDED_DIM
        # squeeze and return to (batchsize=1, dim, length)
        # https://github.com/pytorch/pytorch/issues/1333
        x = torch_nn_func.pad(
            data.permute(0, 2, 1).unsqueeze(2),
            (self.pad_le, self.pad_ri, 0, 0),
            mode=self.pad_mode,
        ).squeeze(2)
        # tanh(conv1())
        # permmute back to (batchsize=1, length, dim)
        output = self.l_ac(self.conv(x))
        return output.permute(0, 2, 1)


#
# Moving average
class MovingAverage(Conv1dKeepLength):
    """Wrapper to define a moving average smoothing layer
    Note: MovingAverage can be implemented using TimeInvFIRFilter too.
          Here we define another Module dicrectly on Conv1DKeepLength
    """

    def __init__(self, feature_dim, window_len, causal=False, pad_mode="replicate"):
        super(MovingAverage, self).__init__(
            feature_dim,
            feature_dim,
            1,
            window_len,
            causal,
            groups=feature_dim,
            bias=False,
            tanh=False,
            pad_mode=pad_mode,
        )
        # set the weighting coefficients
        torch_nn.init.constant_(self.conv.weight, 1 / window_len)
        # turn off grad for this layer
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, data):
        return super(MovingAverage, self).forward(data)


#


# Up sampling
class UpSampleLayer(torch_nn.Module):
    """Wrapper over up-sampling
    Input tensor: (batchsize=1, length, dim)
    Ouput tensor: (batchsize=1, length * up-sampling_factor, dim)
    """

    def __init__(self, feature_dim, up_sampling_factor, smoothing=False):
        super(UpSampleLayer, self).__init__()
        # wrap a up_sampling layer
        self.scale_factor = up_sampling_factor
        self.l_upsamp = torch_nn.Upsample(
            scale_factor=self.scale_factor
        )  # , mode='linear')f0 smoothしないほうが良い
        if smoothing:
            self.l_ave1 = MovingAverage(feature_dim, self.scale_factor)
            self.l_ave2 = MovingAverage(feature_dim, self.scale_factor)
        else:
            self.l_ave1 = torch_nn.Identity()
            self.l_ave2 = torch_nn.Identity()
        return

    def forward(self, x):
        # permute to (batchsize=1, dim, length)
        up_sampled_data = self.l_upsamp(x.permute(0, 2, 1))

        # permute it backt to (batchsize=1, length, dim)
        # and do two moving average
        return self.l_ave1(self.l_ave2(up_sampled_data.permute(0, 2, 1)))


# Neural filter block (1 block)
class NeuralFilterBlock(torch_nn.Module):
    """Wrapper over a single filter block"""

    def __init__(self, signal_size, hidden_size, kernel_size=3, conv_num=10):
        super(NeuralFilterBlock, self).__init__()
        self.signal_size = signal_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv_num = conv_num
        self.dilation_s = [np.power(2, x) for x in np.arange(conv_num)]

        # ff layer to expand dimension
        self.l_ff_1 = torch_nn.Linear(signal_size, hidden_size, bias=False)

        self.l_ff_1_tanh = torch_nn.Tanh()

        # dilated conv layers
        tmp = [
            Conv1dKeepLength(
                hidden_size, hidden_size, x, kernel_size, causal=True, bias=False
            )
            for x in self.dilation_s
        ]
        self.l_convs = torch_nn.ModuleList(tmp)

        # ff layer to de-expand dimension
        self.l_ff_2 = torch_nn.Linear(hidden_size, hidden_size // 4, bias=False)

        self.l_ff_2_tanh = torch_nn.Tanh()
        self.l_ff_3 = torch_nn.Linear(hidden_size // 4, signal_size, bias=False)

        self.l_ff_3_tanh = torch_nn.Tanh()

        # a simple scale
        self.scale = torch_nn.Parameter(torch.tensor([0.1]), requires_grad=False)
        return

    def forward(self, signal, context):
        """
        Assume: signal (batchsize=1, length, signal_size)
                context (batchsize=1, length, hidden_size)
        Output: (batchsize=1, length, signal_size)
        """
        # expand dimension
        tmp_hidden = self.l_ff_1_tanh(self.l_ff_1(signal))

        # loop over dilated convs
        # output of a d-conv is input + context + d-conv(input)
        for l_conv in self.l_convs:
            tmp_hidden = tmp_hidden + l_conv(tmp_hidden) + context

        # to be consistent with legacy configuration in CURRENNT
        tmp_hidden = tmp_hidden * self.scale

        # compress the dimesion and skip-add
        tmp_hidden = self.l_ff_2_tanh(self.l_ff_2(tmp_hidden))
        tmp_hidden = self.l_ff_3_tanh(self.l_ff_3(tmp_hidden))
        output_signal = tmp_hidden + signal

        return output_signal


#
# Sine waveform generator
#
# Sine waveform generator
class SineGen(torch_nn.Module):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)

    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)

    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case
            sines = torch.sin(torch.cumsum(rad_values, dim=1) * 2 * np.pi)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def _f02rosenberg(self, f0_values, t1_ratio=0.4, t2_ratio=0.16):
        rad = f0_values / self.sampling_rate  # normalized frequency(0~2pi ->0~1)
        rad_cum = torch.cumsum(rad, 1)  # rad
        rad_cum = rad_cum - torch.trunc(rad_cum)  # rad within (0, 1)
        rosenberg = torch.zeros_like(rad_cum)
        ind1 = rad_cum < t1_ratio
        ind2 = (rad_cum >= t1_ratio) * (rad_cum < t1_ratio + t2_ratio)
        rosenberg[ind1] = 1.0 - torch.cos(rad_cum[ind1] / t1_ratio * np.pi)
        rosenberg[ind2] = torch.cos((rad_cum[ind2] - t1_ratio) / t2_ratio * np.pi / 2)
        return rosenberg

    def forward(self, f0):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            if self.flag_for_pulse:
                sine_waves = self._f02rosenberg(f0) * self.sine_amp
            else:
                phase_buf = torch.zeros(
                    f0.shape[0], f0.shape[1], self.dim, device=f0.device
                )
                # fundamental component
                phase_buf[:, :, 0] = f0[:, :, 0]
                for idx in np.arange(self.harmonic_num):
                    # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
                    phase_buf[:, :, idx + 1] = phase_buf[:, :, 0] * (idx + 2)

                # generate sine waveforms
                sine_waves = self._f02sine(phase_buf) * self.sine_amp

            # generate uv signal
            # uv = torch.ones(f0.shape)
            # uv = uv * (f0 > self.voiced_threshold)
            uv = self._f02uv(f0)

            # noise: for unvoiced should be similar to sine_amp
            #        std = self.sine_amp/3 -> max value ~ self.sine_amp
            # .       for voiced regions is self.noise_std
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            # first: set the unvoiced part to 0 by uv
            # then: additive noise
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


#####
## Model definition
##

## For condition module only provide Spectral feature to Filter block
class CondModuleBaseNSF(torch_nn.Module):
    """Condition module for hn-sinc-NSF

    Upsample and transform input features
    CondModuleBaseNSF(input_dimension, output_dimension, up_sample_rate,
               blstm_dimension = 64, cnn_kernel_size = 3)

    Spec, F0, cut_off_freq = CondModuleBaseNSF(features, F0)

    Both input features should be frame-level features
    If x doesn't contain F0, just ignore the returned F0

    CondModuleBaseNSF(input_dim, output_dim, up_sample,
                        blstm_s = 64, cnn_kernel_s = 3,
                        voiced_threshold = 0):

    input_dim: sum of dimensions of input features
    output_dim: dim of the feature Spec to be used by neural filter-block
    up_sample: up sampling rate of input features
    blstm_s: dimension of the features from blstm (default 64)
    cnn_kernel_s: kernel size of CNN in condition module (default 3)
    voiced_threshold: f0 > voiced_threshold is voiced, otherwise unvoiced
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        up_sample,
        blstm_s=64,
        cnn_kernel_s=3,
        voiced_threshold=0,
    ):
        super(CondModuleBaseNSF, self).__init__()

        # input feature dimension
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.up_sample = up_sample
        self.blstm_s = blstm_s
        self.cnn_kernel_s = cnn_kernel_s
        self.cut_f_smooth = up_sample * 4
        self.voiced_threshold = voiced_threshold

        # the blstm layer
        self.l_blstm = BLSTMLayer(input_dim, self.blstm_s)

        # the CNN layer (+1 dim for cut_off_frequence of sinc filter)
        self.l_conv1d = Conv1dKeepLength(
            self.blstm_s, self.output_dim, dilation_s=1, kernel_s=self.cnn_kernel_s
        )
        # Upsampling layer for hidden features
        self.l_upsamp = UpSampleLayer(self.output_dim, self.up_sample, True)
        # separate layer for up-sampling normalized F0 values
        self.l_upsamp_f0_hi = UpSampleLayer(1, self.up_sample, True)

        # Upsampling for F0: don't smooth up-sampled F0
        self.l_upsamp_F0 = UpSampleLayer(1, self.up_sample, False)

        # Another smoothing layer to smooth the cut-off frequency
        # for sinc filters. Use a larger window to smooth
        self.l_cut_f_smooth = MovingAverage(1, self.cut_f_smooth)

    def forward(self, feature, f0):
        """spec, f0 = forward(self, feature, f0)
        feature: (batchsize, length, dim)
        f0: (batchsize, length, dim=1), which should be F0 at frame-level

        spec: (batchsize, length, self.output_dim), at wave-level
        f0: (batchsize, length, 1), at wave-level
        """
        tmp = self.l_upsamp(self.l_conv1d(self.l_blstm(feature)))

        # concatenat normed F0 with hidden spectral features
        context = torch.cat(
            (
                tmp[:, :, 0 : self.output_dim - 1],
                self.l_upsamp_f0_hi(feature[:, :, -1:]),
            ),
            dim=2,
        )
        # directly up-sample F0 without smoothing
        f0_upsamp = self.l_upsamp_F0(f0)

        # return
        return context, f0_upsamp


# For source module
class SourceModuleBaseNSF(torch_nn.Module):
    """SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)

    Sine_source, noise_source = SourceModuleBaseNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
        flag_for_pulse=False,
    ):
        super(SourceModuleBaseNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshod,
            flag_for_pulse=flag_for_pulse,
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch_nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch_nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleBaseNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, uv


# For Filter module
class FilterModuleBaseNSF(torch_nn.Module):
    """Filter for Hn-sinc-NSF
    FilterModuleBaseNSF(signal_size, hidden_size, sinc_order = 31,
                          block_num = 5, kernel_size = 3,
                          conv_num_in_block = 10)
    signal_size: signal dimension (should be 1)
    hidden_size: dimension of hidden features inside neural filter block
    sinc_order: order of the sinc filter
    block_num: number of neural filter blocks in harmonic branch
    kernel_size: kernel size in dilated CNN
    conv_num_in_block: number of d-conv1d in one neural filter block

    Usage:
    output = FilterModuleBaseNSF(har_source, noi_source, cut_f, context)
    har_source: source for harmonic branch (batchsize, length, dim=1)
    noi_source: source for noise branch (batchsize, length, dim=1)
    cut_f: cut-off-frequency of sinc filters (batchsize, length, dim=1)
    context: hidden features to be added (batchsize, length, dim)
    output: (batchsize, length, dim=1)
    """

    def __init__(
        self,
        signal_size,
        hidden_size,
        block_num=5,
        kernel_size=3,
        conv_num_in_block=10,
    ):
        super(FilterModuleBaseNSF, self).__init__()
        self.signal_size = signal_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.block_num = block_num
        self.conv_num_in_block = conv_num_in_block

        # filter blocks for harmonic branch
        tmp = [
            NeuralFilterBlock(signal_size, hidden_size, kernel_size, conv_num_in_block)
            for x in range(self.block_num)
        ]
        self.l_har_blocks = torch_nn.ModuleList(tmp)

    def forward(self, har_component, cond_feat):
        """"""
        # harmonic component
        for l_har_block in self.l_har_blocks:
            har_component = l_har_block(har_component, cond_feat)

        # get output
        return har_component


## FOR MODEL
class Model(torch_nn.Module):
    """Model definition"""

    def __init__(self, in_dim, out_dim, args, mean_std=None):
        super(Model, self).__init__()

        torch.manual_seed(1)
        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(
            in_dim, out_dim, args, mean_std
        )
        self.input_mean = torch_nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch_nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch_nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch_nn.Parameter(out_s, requires_grad=False)
        self.input_dim = in_dim
        self.output_dim = out_dim

        # configurations
        # amplitude of sine waveform (for each harmonic)
        self.sine_amp = 0.1
        # standard deviation of Gaussian noise for additive noise
        self.noise_std = 0.003
        # dimension of hidden features in filter blocks
        self.hidden_dim = 64
        # upsampling rate on input acoustic features (16kHz * 5ms = 80)
        self.upsamp_rate = prj_conf.input_reso[1]
        # sampling rate (Hz)
        # TODO:config.pyとconfig.jsonを統合する
        self.sampling_rate = prj_conf.wav_samp_rate
        # CNN kernel size in filter blocks
        self.cnn_kernel_s = 3
        # number of filter blocks (for harmonic branch)
        # noise branch only uses 1 block
        self.filter_block_num = 5
        # number of dilated CNN in each filter block
        self.cnn_num_in_block = 10
        # number of harmonic overtones in source
        self.harmonic_num = 9

        # upsample mel to match f0 length
        if prj_conf.input_reso[0] != prj_conf.input_reso[1]:
            self.mel_up = UpSampleLayer(
                80, prj_conf.input_reso[0] / prj_conf.input_reso[1]
            )
        else:
            self.mel_up = None

        # the three modules
        self.m_cond = CondModuleBaseNSF(
            self.input_dim,
            self.hidden_dim,
            self.upsamp_rate,
            cnn_kernel_s=self.cnn_kernel_s,
        )

        self.m_source = SourceModuleBaseNSF(
            self.sampling_rate, self.harmonic_num, self.sine_amp, self.noise_std
        )

        self.m_filter = FilterModuleBaseNSF(
            self.output_dim,
            self.hidden_dim,
            self.filter_block_num,
            self.cnn_kernel_s,
            self.cnn_num_in_block,
        )

    def prepare_mean_std(self, in_dim, out_dim, args, data_mean_std=None):
        """"""
        if data_mean_std is not None:
            in_m = torch.from_numpy(data_mean_std[0])
            in_s = torch.from_numpy(data_mean_std[1])
            out_m = torch.from_numpy(data_mean_std[2])
            out_s = torch.from_numpy(data_mean_std[3])
            if in_m.shape[0] != in_dim or in_s.shape[0] != in_dim:
                print("Input dim: {:d}".format(in_dim))
                print("Mean dim: {:d}".format(in_m.shape[0]))
                print("Std dim: {:d}".format(in_s.shape[0]))
                print("Input dimension incompatible")
                sys.exit(1)
            if out_m.shape[0] != out_dim or out_s.shape[0] != out_dim:
                print("Output dim: {:d}".format(out_dim))
                print("Mean dim: {:d}".format(out_m.shape[0]))
                print("Std dim: {:d}".format(out_s.shape[0]))
                print("Output dimension incompatible")
                sys.exit(1)
        else:
            in_m = torch.zeros([in_dim])
            in_s = torch.zeros([in_dim])
            out_m = torch.zeros([out_dim])
            out_s = torch.zeros([out_dim])

        return in_m, in_s, out_m, out_s

    def normalize_input(self, x):
        """normalizing the input data"""
        return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        """normalizing the target data"""
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y):
        """denormalizing the generated output from network"""
        return y * self.output_std + self.output_mean

    def forward(self, mel, f0):
        """definition of forward method
        Assume x (batchsize=1, length, dim)
        Return output(batchsize=1, length)
        """
        # upsample mel to match f0 length
        if self.mel_up is not None:
            mel = self.mel_up(mel)[:, : f0.size(1), :]
        # normalize the input features data
        feat = self.normalize_input(torch.cat([mel, f0], dim=2))

        # condition module
        # feature-to-filter-block, f0-up-sampled, cut-off-f-for-sinc,
        #  hidden-feature-for-cut-off-f
        cond_feat, f0_upsamped = self.m_cond(feat, f0)

        # source module
        # harmonic-source, noise-source (for noise branch), uv
        har_source, uv = self.m_source(f0_upsamped)

        # neural filter module
        # output
        output = self.m_filter(har_source, cond_feat)

        return output.squeeze(-1)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
                torch.nn.utils.weight_norm(m, dim=None)

        self.apply(_apply_weight_norm)


class MelGANDiscriminator(torch.nn.Module):
    """MelGAN discriminator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_sizes=[5, 3],
        channels=16,
        max_downsample_channels=1024,
        bias=True,
        downsample_scales=[4, 4, 4, 4],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
    ):
        """Initilize MelGAN discriminator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15,
                the last two layers' kernel size will be 5 and 3, respectively.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
        """
        super(MelGANDiscriminator, self).__init__()
        self.layers = torch.nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        # add first layer
        self.layers += [
            torch.nn.Sequential(
                getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params),
                torch.nn.Conv1d(
                    in_channels, channels, np.prod(kernel_sizes), bias=bias
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers
        in_chs = channels
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chs,
                        out_chs,
                        kernel_size=downsample_scale * 10 + 1,
                        stride=downsample_scale,
                        padding=downsample_scale * 5,
                        groups=in_chs // 4,
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_chs,
                    out_chs,
                    kernel_sizes[0],
                    padding=(kernel_sizes[0] - 1) // 2,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            torch.nn.Conv1d(
                out_chs,
                out_channels,
                kernel_sizes[1],
                padding=(kernel_sizes[1] - 1) // 2,
                bias=bias,
            ),
        ]

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            List: List of output tensors of each layer.
        """
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]

        return outs


class MelGANMultiScaleDiscriminator(torch.nn.Module):
    """MelGAN multi-scale discriminator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        scales=3,
        downsample_pooling="AvgPool1d",
        # follow the official implementation setting
        downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
            "count_include_pad": False,
        },
        kernel_sizes=[5, 3],
        channels=16,
        max_downsample_channels=1024,
        bias=True,
        downsample_scales=[4, 4, 4, 4],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
        use_weight_norm=True,
    ):
        """Initilize MelGAN multi-scale discriminator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            kernel_sizes (list): List of two kernel sizes. The sum will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_causal_conv (bool): Whether to use causal convolution.
        """
        super(MelGANMultiScaleDiscriminator, self).__init__()
        self.discriminators = torch.nn.ModuleList()

        # add discriminators
        for _ in range(scales):
            self.discriminators += [
                MelGANDiscriminator(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    channels=channels,
                    max_downsample_channels=max_downsample_channels,
                    bias=bias,
                    downsample_scales=downsample_scales,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                    pad=pad,
                    pad_params=pad_params,
                )
            ]
        self.pooling = getattr(torch.nn, downsample_pooling)(
            **downsample_pooling_params
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.
        """
        outs = []
        for f in self.discriminators:
            outs += [f(x)]
            x = self.pooling(x)

        return outs

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.
        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py
        """

        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                m.weight.data.normal_(0.0, 0.02)

        self.apply(_reset_parameters)


class Loss:
    """Wrapper to define loss function"""

    def __init__(self, device):
        """"""
        # frame shift (number of points)
        self.frame_hops = [80, 40, 640]
        # frame length
        self.frame_lens = [320, 80, 1920]
        # fft length
        self.fft_n = [512, 128, 2048]
        # window type in stft
        self.win = torch.hann_window
        # floor in log-spectrum-amplitude calculating
        self.amp_floor = 0.00001
        # loss function
        self.loss = torch_nn.MSELoss()
        self.gan_loss = torch_nn.MSELoss()  # BCELoss()

        self.device = device

    def stft_loss(self, outputs, target):
        """Loss().compute(outputs, target) should return
        the Loss in torch.tensor format
        Assume output and target as (batchsize=1, length)
        """

        # convert from (batchsize=1, length, dim=1) to (1, length)
        if target.ndim == 3:
            target.squeeze_(-1)

        # compute loss
        loss = 0
        for frame_shift, frame_len, fft_p in zip(
            self.frame_hops, self.frame_lens, self.fft_n
        ):

            x_stft = torch.stft(
                outputs,
                fft_p,
                frame_shift,
                frame_len,
                window=self.win(frame_len, device=self.device),
                onesided=True,
                pad_mode="constant",
                return_complex=True,
            )
            y_stft = torch.stft(
                target,
                fft_p,
                frame_shift,
                frame_len,
                window=self.win(frame_len, device=self.device),
                onesided=True,
                pad_mode="constant",
                return_complex=True,
            )
            x_stft = torch.abs(x_stft).pow(2)
            y_stft = torch.abs(y_stft).pow(2)

            # spectral convergence
            # loss += torch.norm(y_stft - x_stft) / torch.norm(y_stft)

            # log STFT magnitude loss
            x_sp_amp = torch.log(x_stft + self.amp_floor)
            y_sp_amp = torch.log(y_stft + self.amp_floor)
            loss += self.loss(x_sp_amp, y_sp_amp)

        return loss / 3

    def adversarial_loss(self, fake):
        adv_loss = 0
        for scale in fake:
            adv_loss += -scale[-1].mean()
        return adv_loss  # self.gan_loss(fake, fake.new_ones(fake.size()))

    def discriminator_loss(self, real, fake):
        real_loss = 0
        fake_loss = 0
        for r_scale, f_scale in zip(real, fake):
            real_loss += torch_nn_func.relu(1 - r_scale[-1]).mean()
            fake_loss += torch_nn_func.relu(1 + f_scale[-1]).mean()
        return real_loss, fake_loss


if __name__ == "__main__":
    from torchsummary import summary

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Model(in_dim=81, out_dim=1, args=None).to(device)
    discriminator = MelGANMultiScaleDiscriminator().to(device)

    mel = torch.randn(1, 32000, 80)
    f0 = torch.randn(1, 32000, 1)
    summary(model, mel, f0)
