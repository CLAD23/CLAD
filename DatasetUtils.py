'''
Datasets and augmentations used in CLAD. This code is built on RawNet2 and ASVspoof 2021 Baseline repository.
'''

import os
import random
import torch
import torchaudio
import TorchTimeStretch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import librosa  # used by ASVspoof official data utils


class AddWhiteNoise(nn.Module):
    def __init__(self, max_snr_db, min_snr_db):
        super(AddWhiteNoise, self).__init__()
        self.max_snr_db = max_snr_db
        self.min_snr_db = min_snr_db
    def add_white_noise(self, audio, max_snr_db, min_snr_db):
        # Generate a random SNR between 0 and the maximum value
        snr_db = random.uniform(min_snr_db, max_snr_db)
        # Calculate the signal power
        signal_power = torch.mean(audio ** 2, dim=-1, keepdim=True)
        # Convert the SNR from dB to a power ratio
        snr = 10 ** (snr_db / 10)
        # Calculate the noise power
        noise_power = signal_power / snr
        # Generate white noise with the same shape as the input audio
        noise = torch.randn_like(audio) * torch.sqrt(noise_power)
        # Add the noise to the audio
        noisy_audio = audio + noise
        return noisy_audio
    def forward(self, audio, max_snr_db=None, min_snr_db=None):
        if max_snr_db == None:
            max_snr_db = self.max_snr_db
        if min_snr_db == None:
            min_snr_db = self.min_snr_db
        return self.add_white_noise(audio, max_snr_db, min_snr_db)


class VolumeChange(nn.Module):
    def __init__(self, max_vol, min_vol):
        super(VolumeChange, self).__init__()
        self.max_vol = max_vol
        self.min_vol = min_vol       

    def change_volume(self, audio, max_vol, min_vol):
        vol_gain = random.uniform(min_vol, max_vol)
        vol_transform = torchaudio.transforms.Vol(gain=vol_gain,gain_type='amplitude')
        return vol_transform(audio)
    def forward(self, audio, max_vol=None, min_vol=None):
        if max_vol == None:
            max_vol = self.max_vol
        if min_vol == None:
            min_vol = self.min_vol
        return self.change_volume(audio, max_vol, min_vol)

class AddFade(nn.Module):
    def __init__(self, max_fade_size=.5, fade_shape=None, fix_fade_size=False):
        super(AddFade, self).__init__()
        self.max_fade_size = max_fade_size
        self.fade_shape = fade_shape
        self.fix_fade_size = fix_fade_size
    def add_fade(self, audio, fade_in_len, fade_out_len, fade_shape):
        fade_transform = torchaudio.transforms.Fade(fade_in_len=fade_in_len, fade_out_len=fade_out_len, fade_shape=fade_shape)
        return fade_transform(audio)
    def forward(self, audio, fade_in_len=None, fade_out_len=None, fade_shape=None):
        wave_length = audio.size()[-1]
        # wave_length = audio.shape[1] 
        if fade_in_len == None:
            if self.fix_fade_size:
                fade_in_len = int(self.max_fade_size  * wave_length)
            else:
                fade_in_len = random.randint(0, int(self.max_fade_size  * wave_length))
        if fade_out_len == None:
            if self.fix_fade_size:
                fade_out_len = int(self.max_fade_size  * wave_length)
            else:
                fade_out_len = random.randint(0, int(self.max_fade_size  * wave_length))
        if fade_shape == None:
            if self.fade_shape == None:
                fade_shape = random.choice(["quarter_sine", "half_sine", "linear", "logarithmic", "exponential"])
            else:
                fade_shape = self.fade_shape
        return self.add_fade(audio, fade_in_len, fade_out_len, fade_shape)

class WaveTimeStretch(nn.Module):
    def __init__(self, max_ratio, min_ratio, sample_rate=16000, n_fft=0):
        super(WaveTimeStretch, self).__init__()
        self.max_ratio = max_ratio
        self.min_ratio = min_ratio
        self.sample_rate = sample_rate
        self.n_fft = n_fft
    def time_strech(self, audio, max_ratio, min_ratio, n_fft):
        stretch_ratio = random.uniform(min_ratio, max_ratio)
        return TorchTimeStretch.time_stretch(input=audio, stretch=stretch_ratio, sample_rate=self.sample_rate, n_fft=n_fft)
    def forward(self, audio, max_ratio=None, min_ratio=None, n_fft=None):
        if max_ratio == None:
            max_ratio = self.max_ratio
        if min_ratio == None:
            min_ratio = self.min_ratio
        if n_fft == None:
            n_fft = self.n_fft
        return self.time_strech(audio, max_ratio, min_ratio, n_fft)

# Codec manipulation which is not introduced in the paper. This manipulation have little impact and have to be used on CPU.
class CodecApply(nn.Module):
    def __init__(self, codec=None, sample_rate=16000):
        super(CodecApply, self).__init__()
        self.codec = codec
        self.sample_rate = sample_rate

    def codec_apply(self, audio, codec):
        assert codec in ['ALAW', 'ULAW'] , "codec must be in ['ALAW', 'ULAW']"
        return torchaudio.functional.apply_codec(waveform=audio, sample_rate=self.sample_rate, format="wav", encoding=codec).squeeze(0)
    def forward(self, audio, codec=None):
        # if the audio only have 1 dim, add a channel dim
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        encoding_list = ['ALAW', 'ULAW']
        if codec == None:
            codec = self.codec
        if codec == None:
            codec = random.choice(encoding_list)
        return self.codec_apply(audio, codec)

class AddEnvironmentalNoise(nn.Module):
    def __init__(self, max_snr_db, min_snr_db, noise_dataset_path, device=None, noise_path=None, noise_category=None, audio_len=64600, add_before_audio_len = None, sample_rate=16000):
        super(AddEnvironmentalNoise, self).__init__()
        self.noise_path = noise_path
        self.max_snr_db = max_snr_db
        self.min_snr_db = min_snr_db
        self.noise_category = noise_category
        self.audio_len = audio_len
        self.device = device
        self.sample_rate = sample_rate
        self.add_before_audio_len = add_before_audio_len
        # predefined noise
        self.noise_filename_dict = {
            "wind": "1-29532-A-16.wav",
            "footsteps": "1-155858-D-25.wav",
            "breathing": "2-98392-A-23.wav",
            "coughing": "4-157296-A-24.wav",
            "rain": "3-157615-A-10.wav",
            "clock_tick": "5-209833-A-38.wav",
            "sneezing": "4-167642-A-21.wav"
        }
        self.noise_dataset_path = noise_dataset_path
        # if the noise_path is given, just load the noise and calculate power
        # elif the noise_category is given, get the noise_path from noise_filename_dict and load the noise and calculate power
        # else(the training strategy) randomly choose a noise category and load the noise and calculate power
        if self.noise_path==None:
            if self.noise_category == None:
                # preload the noise, noise tensor power for each category and store them in a dict
                self.noise_dict = {}
                for noise_category in self.noise_filename_dict.keys():
                    noise_path = os.path.join(self.noise_dataset_path, self.noise_filename_dict[noise_category])
                    noise_tensor, noise_tensor_power = self.load_noise_and_power(noise_path)
                    self.noise_dict[noise_category] = (noise_tensor, noise_tensor_power)
                self.noise_tensor = None
            else:
                self.noise_path = os.path.join(self.noise_dataset_path, self.noise_filename_dict[self.noise_category])
                self.noise_tensor, self.noise_tensor_power = self.load_noise_and_power(self.noise_path)
        else:
            self.noise_tensor, self.noise_tensor_power = self.load_noise_and_power(self.noise_path)
            print(f"Add Environmental Noise Augmentation Initialized. Noise path: {self.noise_path}, Noise shape {self.noise_tensor.shape}, Noise power {self.noise_tensor_power}")
        # assert self.noise_tensor_power > 0 , "Check the noise audio, the power of it should be positive."

    def load_noise_and_power(self, noise_path):
        noise_tensor, noise_sr = torchaudio.load(noise_path)
        noise_tensor = noise_tensor.squeeze(0)  # remove the channel dimension.
        if self.device != None:
            noise_tensor = noise_tensor.to(self.device)
        noise_tensor = torchaudio.functional.resample(noise_tensor, noise_sr, self.sample_rate)  # The ESC-50 dataset have a 44.1kHz sr.
        if noise_tensor.shape[-1] < self.audio_len:
            noise_tensor = noise_tensor.repeat(self.audio_len // noise_tensor.shape[-1] + 1)[:self.audio_len]
        else:
            noise_tensor = noise_tensor[:self.audio_len]
        noise_tensor_power = torch.mean(noise_tensor ** 2, dim=-1, keepdim=True)
        return noise_tensor, noise_tensor_power


    def add_environmental_noise(self, audio, max_snr_db, min_snr_db):
        snr_db = random.uniform(min_snr_db, max_snr_db)
        snr = 10 ** (snr_db / 10)
        signal_power = torch.mean(audio ** 2, dim=-1, keepdim=True)
        noise_power_needed = signal_power / snr
        noise_power_factor = noise_power_needed / self.noise_tensor_power
        noise = self.noise_tensor * torch.sqrt(noise_power_factor)
        if self.add_before_audio_len != None:
            if len(audio.shape) ==1:
                tmp_zeros = torch.zeros_like(audio[:self.add_before_audio_len])
            else:
                tmp_zeros = torch.zeros_like(audio[:, :self.add_before_audio_len])
            audio = torch.cat([tmp_zeros, audio], dim=-1)
            noise = torch.cat([noise, tmp_zeros], dim=-1)
        # Add the noise to the audio
        noisy_audio = audio + noise
        return noisy_audio
    def forward(self, audio, max_snr_db=None, min_snr_db=None):
        if max_snr_db == None:
            max_snr_db = self.max_snr_db
        if min_snr_db == None:
            min_snr_db = self.min_snr_db
        if self.noise_category == None and self.noise_path == None:
            noise_category = random.choice(list(self.noise_dict.keys()))
            self.noise_tensor, self.noise_tensor_power = self.noise_dict[noise_category]
        return self.add_environmental_noise(audio, max_snr_db, min_snr_db)


class ResampleAugmentation(nn.Module):
    '''
    Since the resample takes a lot of time, we setup some predefined resample rate and randomly choose one of them to speed up.
    '''
    def __init__(self, resample_rate: list, original_sr: int=16000, device="cuda"):
        super(ResampleAugmentation, self).__init__()
        self.resample_rate = resample_rate
        self.original_sr = original_sr   
        self.device = device  
        # create the resample augmentations here to speed up the forward process  
        self.resample_transforms = nn.ModuleList()
        for resample_rate in self.resample_rate:
            self.resample_transforms.append(torchaudio.transforms.Resample(orig_freq=self.original_sr, new_freq=resample_rate).to(device))
    def forward(self, audio):
        # choose a random resample transform from self.resample_transforms
        resample_transform = random.choice(self.resample_transforms)
        return resample_transform(audio)
    
class SmoothingAugmentation(nn.Module):
    def __init__(self):
        super(SmoothingAugmentation, self).__init__()
    def set_even_to_avg(self, t:torch.Tensor):
        input_len_odd = t.shape[-1] % 2 == 1
        if len(t.shape) == 1:
            if input_len_odd:
                t = torch.cat((t, torch.tensor([t[-1]])))
            assert t.shape[-1] > 3, "The input is too short!"
            t[2:-1:2] = (t[1:-2:2] + t[3::2]) / 2
            if input_len_odd:
                t = t[:-1]
        elif len(t.shape) == 2:
            if input_len_odd:
                last_column = t[:, -1:]
                t = torch.cat((t, last_column), dim=1)
            t[:,2:-1:2] = (t[:,1:-2:2] + t[:,3::2]) / 2
            if input_len_odd:
                t = t[:,:-1]
        return t
    def forward(self, audio):
        return self.set_even_to_avg(audio)

class AddEchoes(nn.Module):
    def __init__(self, max_delay, min_delay, max_strengh, min_strength):
        super(AddEchoes, self).__init__()
        self.max_delay = max_delay
        self.max_strengh = max_strengh
        self.min_delay = min_delay
        self.min_strength = min_strength
    def add_echoes(self, audio, echo_delay, echo_strengh):
        tmp_audio = audio.clone()
        if len(audio.shape) == 1:
            tmp_audio[echo_delay:] += tmp_audio[:-echo_delay] * echo_strengh
        elif len(audio.shape) == 2:
            tmp_audio[:,echo_delay:] += tmp_audio[:,:-echo_delay] * echo_strengh
        return tmp_audio
    def forward(self, audio, echo_delay=None, echo_strengh=None):
        if echo_delay is not None and echo_strengh is not None:
            return self.add_echoes(audio, echo_delay, echo_strengh)
        else:
            echo_delay = random.randint(self.min_delay, self.max_delay)
            echo_strengh = random.uniform(self.min_strength, self.max_strengh)
        return self.add_echoes(audio, echo_delay, echo_strengh)

class TimeShift(nn.Module):
    def __init__(self, max_shift, min_shift):
        super(TimeShift, self).__init__()
        self.max_shift = max_shift
        self.min_shift = min_shift       

    def time_shift(self, audio:torch.Tensor, shift_len):
        
        return audio.roll(shifts=shift_len, dims=-1)
    def forward(self, audio, max_shift=None, min_shift=None):
        if max_shift == None:
            max_shift = self.max_shift
        if min_shift == None:
            min_shift = self.min_shift
        shift_len = random.randint(min_shift, max_shift)
        return self.time_shift(audio, shift_len)

class AddZeroPadding(nn.Module):
    def __init__(self, max_left_len, min_left_len, max_right_len, min_right_len):
        super(AddZeroPadding, self).__init__()
        self.max_left_len = max_left_len
        self.min_left_len = min_left_len
        self.max_right_len = max_right_len
        self.min_right_len = min_right_len

    def add_zero_padding(self, audio, left_len, right_len):
        if len(audio.shape) ==1:
            left_zeros = torch.zeros([left_len], device=audio.device)
            right_zeros = torch.zeros([right_len], device=audio.device)
        else:
            batch_size = audio.shape[0]
            left_zeros = torch.zeros([batch_size, left_len], device=audio.device)
            right_zeros = torch.zeros([batch_size, right_len], device=audio.device)
        audio = torch.cat([left_zeros, audio, right_zeros], dim=-1)
        return audio

    def forward(self, audio, left_len=None, right_len=None):
        if left_len == None:
            left_len = random.randint(self.min_left_len, self.max_left_len)
        if right_len == None:
            right_len = random.randint(self.min_right_len, self.max_right_len)
        return self.add_zero_padding(audio, left_len, right_len)


# Datasets
class MoCoAudioDataset(torch.utils.data.Dataset):
    '''
    A class for MoCo audio datasets
    '''
    def __init__(self, root_dir, file_list, label_list, transform=None, sample_rate=16000, audio_len=64600):
        '''
        Args:
            root_dir (string): Directory with all the audio files.
            file_list (list): List of audio file names.
            label_list(dict): Dict of the labels for the file names.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        '''        
        self.root_dir = root_dir
        self.file_list = file_list
        self.label_list = label_list
        self.transform = transform
        self.sample_rate = sample_rate
        self.audio_len = audio_len

    def pad_or_clip(self, audio):
        '''
        Pad or randomly clip the audio to make it of length self.audio_len
        '''
        if audio.shape[-1] < self.audio_len:
            audio = audio.repeat(self.audio_len // audio.shape[-1] + 1)[:self.audio_len]
        elif audio.shape[-1] > self.audio_len:
            # randomly clip the audio
            start = random.randint(0, audio.shape[-1] - self.audio_len)
            # start = 0 # exp5 test
            audio = audio[start:start+self.audio_len]
        return audio

    def __getitem__(self, index):
        key =  self.file_list[index]+'.flac'
        audio = torchaudio.load(self.root_dir+key)[0]  # torchaudio.load returns a tuple (tensor, sample_rate)
        audio = audio.squeeze(0)  # Remove the channel dimension
        # pad or clip to make the audio length to self.audio_len
        audio1 = self.pad_or_clip(audio)
        audio2 = self.pad_or_clip(audio)

        # Apply input transformation on CPU, the augmentations on GPU will be applied during training process.
        if self.transform:
            audio1 = self.transform(audio1)
            audio2 = self.transform(audio2)

        key = key[:-5]  # since the key have been added the .flac suffix, we remove it here by take out the last 5 chars
        label = self.label_list[key]
        # Return multiple views of the audio data
        return [audio1, audio2, label]

    def __len__(self):
        return len(self.file_list)
    

# Evaluation utilities
# Mainly modified from https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-RawNet2/data_utils.py by "Hemlata Tak"
# Obtain speaker information for SAMO implementation
def genSpoof_list( dir_meta,is_train=False,is_eval=False):
    utt2spk = {}
    d_meta = {}
    file_list=[]
    with open(dir_meta, 'r') as f:
         l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            spk, key,_,_,label = line.strip().split(' ')
            utt2spk[key] = spk
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list,utt2spk
    
    elif(is_eval):
        for line in l_meta:
            key= line.strip()
            file_list.append(key)
        return file_list
    else:  # so same as is_train ?? by haulyn5
        for line in l_meta:
            spk, key,_,_,label = line.strip().split(' ')
            utt2spk[key] = spk
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta,file_list,utt2spk


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x	
			

class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, utt2spk, base_dir, cut_length=64600):
            '''self.list_IDs	: list of strings (each string: utt key),
               self.labels      : dictionary (key: utt key, value: label integer)'''
               
            self.list_IDs = list_IDs
            self.labels = labels
            self.base_dir = base_dir
            self.cut = cut_length
            self.utt2spk = utt2spk
        
    def __len__(self):
           return len(self.list_IDs)


    def __getitem__(self, index):
            # self.cut=64600 # take ~4 sec audio (64600 samples)
            key = self.list_IDs[index]
            X,fs = librosa.load(self.base_dir+'flac/'+key+'.flac', sr=16000) 
            X_pad= pad(X,self.cut)
            x_inp= torch.Tensor(X_pad)
            x_inp = torch.unsqueeze(x_inp, 0)  # added by haulyn5, add a dimension for channels In order to be consistent with the previous dataset
            y = self.labels[key]
            spk = self.utt2spk[key]
            return x_inp, spk, y

# A upgraded version of pad_or_clip function, which can process batched audio, zero padding or  clipping them to the same length
def pad_or_clip_batch(audio, audio_len, random_clip=True):
        '''
        Pad or randomly clip the audio to make it of length audio_len
        '''
        if audio.shape[-1] < audio_len:
            audio = torch.nn.functional.pad(audio, (0, audio_len - audio.shape[-1]))
        elif audio.shape[-1] > audio_len:
            if random_clip == True:
                # randomly clip the audio
                start = random.randint(0, audio.shape[-1] - audio_len)
            else:
                start = 0 # clip from the beginning, which is the standard implementation of AASIST and RawNet2
                audio = audio[:, start:start+ audio_len]
        return audio