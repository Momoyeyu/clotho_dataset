######################################################
#                                                    #
# It is a demo of using Clotho datasets for training #
#                                                    #
######################################################


from torch.utils.data import DataLoader
from clotho import ClothoDataset
from hparam import hparam as hp
from preprocess import collate_fn
import matplotlib.pyplot as plt
import numpy as np


# init datasets
development_dataset = ClothoDataset(hp.preprocess.dev_out)
evaluation_dataset = ClothoDataset(hp.preprocess.eval_out)

# create DataLoader
development_loader = DataLoader(
    development_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)
evaluation_loader = DataLoader(
    evaluation_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

# test loader
for audios_padded, captions in development_loader:
    audio_sample = audios_padded[0].numpy()
    caption_sample = captions[0]

    num_samples = len(audio_sample)
    duration = num_samples / hp.data.sr  # cal second
    time_axis = np.linspace(0, duration, num=num_samples)

    # plot
    plt.figure(figsize=(14, 5))
    plt.plot(time_axis, audio_sample)
    plt.title(f'Audio Waveform and Caption:\n"{caption_sample}"')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.show()
    break
