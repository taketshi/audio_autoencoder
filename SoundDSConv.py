import torch
import torchaudio
from torchaudio import transforms
import random
from torch.utils.data import Dataset

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDSConv(Dataset):
    def __init__(self, df, sr = 8000, duration = 8):
        self.df = df
        self.duration = duration*1000 # 8 seconds
        self.sr = sr
        self.channel = 1
        self.spec_size = None

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)
    
    # ----------------------------
    # Choose sample duration
    # ----------------------------
    def set_sample_duration(self, duration):
        self.duration = duration
    
    # ----------------------------
    # Choose sample rate
    # ----------------------------
    def set_sample_rate(self, sample_rate):
        self.sr = sample_rate
    
    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        aud = torchaudio.load(self.df.loc[idx]['path_to_data'])
        waveform, SAMPLE_RATE = aud

        waveform = waveform[0] # mix down

        spectrogram = transforms.Spectrogram(n_fft=1024,
                                             hop_length=512)

        spec = spectrogram(waveform)

        if self.spec_size == None:
            self.spec_size = spec.shape

        return spec.reshape(1, self.spec_size[0], self.spec_size[1])
