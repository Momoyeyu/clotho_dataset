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

        # 加载音频和字幕
        audio = np.load(os.path.join(self.npy_dir, audio_file))
        captions = np.load(os.path.join(self.npy_dir, captions_file), allow_pickle=True)

        if self.transform:
            audio = self.transform(audio)

        # 随机选择一条字幕
        caption = np.random.choice(captions)

        return audio, caption  # 返回NumPy数组即可，在collate_fn中转换为张量
