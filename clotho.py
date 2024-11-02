import os
import numpy as np
from torch.utils.data import Dataset


class ClothoDataset(Dataset):
    def __init__(self, npy_dir, transform=None):
        self.npy_dir = npy_dir
        self.transform = transform
        self.audio_files = [f for f in os.listdir(npy_dir) if f.endswith('_audio.npy')]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        captions_file = audio_file.replace('_audio.npy', '_captions.npy')
        audio = np.load(os.path.join(self.npy_dir, audio_file))
        captions = np.load(os.path.join(self.npy_dir, captions_file), allow_pickle=True)
        if self.transform:
            audio = self.transform(audio)
        caption = np.random.choice(captions)
        return audio, caption
