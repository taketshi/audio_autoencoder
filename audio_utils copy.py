import torch
import torchaudio
from torchaudio import transforms
from torch.utils.data import DataLoader, Dataset
import random



class AudioUtil():
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def load(audio_file, normalize=False):
        sig, sr = torchaudio.load(audio_file, normalize=normalize)
        return (sig, sr)
    
    # ----------------------------
    # Convert the given audio to the desired number of channels
    # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel=1):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
          # Nothing to do
          return aud

        if (new_channel == 1):
          # Convert from stereo to mono by selecting only the first channel
          resig = sig[:1, :]
        else:
          # Convert from mono to stereo by duplicating the first channel
          resig = torch.cat([sig, sig])

        return ((resig, sr))
    
    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
          # Nothing to do
          return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))
    
    # ----------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
          # Truncate the signal to the given length
          sig = sig[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)
    
    # ----------------------------
    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)
    
    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_fft=4*1024, hop_len=int(512/2)):
        waveform, _ = aud
        #top_db = 80

        spectrogram = transforms.Spectrogram(n_fft=n_fft,
                                    hop_length=hop_len,
                                    center=True,
                                    pad_mode="reflect",
                                    power=2.0)

        spec = spectrogram(waveform)
        
        # Convert spectrogram to decibel (dB) units
        # amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        # spectrogram_db = amplitude_to_db(spec)
        
        return spec #spectrogram_db
        
    # ----------------------------
    # Recover the Waveform
    # ----------------------------
    @staticmethod
    def recover_waveform(spec, n_fft=4*1024, hop_len=int(512/2)):
        griffin_lim = transforms.GriffinLim(n_fft=n_fft,
                           hop_length=hop_len)

        rec_waveform = griffin_lim(spec)
        
        return rec_waveform



# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, df):
        self.df = df
        self.duration = 50000
        self.sr = 44100
        self.channel = 1
        self.shift_pct = 0.4

    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)

    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.df.loc[idx]['path_to_data']
        aud = AudioUtil.load(audio_file)
        # Some sounds have a higher sample rate, or fewer channels compared to the
        # majority. So make all sounds have the same number of channels and same
        # sample rate. Unless the sample rate is the same, the pad_trunc will still
        # result in arrays of different lengths, even though the sound duration is
        # the same.
        rechan = AudioUtil.rechannel(aud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, max_ms=self.duration)
        
        sgram = AudioUtil.spectro_gram(dur_aud, n_fft=int(1024), hop_len=int(512))

        return sgram
