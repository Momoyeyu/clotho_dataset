import pandas as pd
import numpy as np
from librosa import load
import torch
import os
from hparam import hparam as hp
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    func for padding

    :param batch: (audio, caption)
    :return audios_padded: (batch_size, max_length)
    :return captions: a list of captioning
    """
    # 将音频数据转换为张量，并记录每个音频的长度
    audios = [torch.tensor(sample[0]) for sample in batch]
    captions = [sample[1] for sample in batch]

    # 使用pad_sequence进行填充
    audios_padded = pad_sequence(audios, batch_first=True, padding_value=0)

    return audios_padded, captions


def preprocess_data(csv_file, audio_dir, output_dir, sr=hp.data.sr, max_length=hp.data.max_length):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        file_name = row['file_name']
        captions = row[['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].values

        # load audio files
        audio_path = os.path.join(audio_dir, file_name)
        audio, _ = load(audio_path, sr=sr)

        # padding or cutting
        if max_length is not None:
            if len(audio) > max_length:
                audio = audio[:max_length]
            else:
                # padding
                padding = np.zeros(max_length - len(audio))
                audio = np.concatenate((audio, padding))

        # save as .npy file
        np.save(os.path.join(output_dir, f"{file_name}_audio.npy"), audio)
        np.save(os.path.join(output_dir, f"{file_name}_captions.npy"), captions)


if __name__ == '__main__':
    print('[INFO] start preprocessing ...')
    preprocess_data(hp.preprocess.dev_cfg, hp.preprocess.dev_path, hp.preprocess.dev_out)
    preprocess_data(hp.preprocess.eval_cfg, hp.preprocess.eval_path, hp.preprocess.eval_out)
    print('[INFO] preprocessing completed.')
