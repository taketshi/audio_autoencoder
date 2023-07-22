import torch
import torchaudio
from torchaudio import transforms
import random
from torch.utils.data import Dataset

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, df):
        self.df = df
        self.duration = 10000 # 10 seconds
        self.sr = 8000
        self.channel = 1

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
    def set_sample_duration(self, sample_rate):
        self.sr = sample_rate
    
    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        aud = torchaudio.load(self.df.loc[idx]['path_to_data'])
        waveform, SAMPLE_RATE = aud

        max_ms = self.duration
        max_len = SAMPLE_RATE//1000 * max_ms
        num_rows, sig_len = waveform.shape

        # padding or truncating 
        if (sig_len > max_len):
          # Truncate the signal to the given length
          waveform = waveform[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            waveform = torch.cat((pad_begin, waveform, pad_end), 1)

        waveform = waveform[0] # mix down

        # Reduce the Sample Rate
        resampler = transforms.Resample(orig_freq=SAMPLE_RATE, new_freq=self.sr)
        waveform = resampler(waveform)

        spectrogram = transforms.Spectrogram(n_fft=1024,
                                             hop_length=512)

        spec = spectrogram(waveform).flatten()

        return spec
