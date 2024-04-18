import os
import numpy as np
import torch
from torch.utils.data import Dataset


def audio_to_mel(path, n_mels=128, n_fft=2048, hop_length=512):
    import librosa
    y, sr = librosa.load(path, sr=None)
    mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


class AudioDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.files = os.listdir(directory)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.directory, self.files[idx])
        mel_spectrogram = audio_to_mel(path)
        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)
        return torch.tensor(mel_spectrogram, dtype=torch.float32)
