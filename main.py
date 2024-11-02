from torch.utils.data import DataLoader
from clotho import ClothoDataset
from hparam import hparam as hp
from preprocess import collate_fn
import matplotlib.pyplot as plt
import numpy as np


# 初始化数据集
development_dataset = ClothoDataset(hp.preprocess.dev_out)
evaluation_dataset = ClothoDataset(hp.preprocess.eval_out)

# 创建DataLoader
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

# 测试数据加载器
for audios_padded, captions in development_loader:
    # 选择第一个音频样本
    audio_sample = audios_padded[0].numpy()  # 转换为NumPy数组
    caption_sample = captions[0]            # 获取对应的字幕

    # 生成时间轴
    num_samples = len(audio_sample)
    duration = num_samples / hp.data.sr  # 计算音频时长（秒）
    time_axis = np.linspace(0, duration, num=num_samples)

    # 绘制波形图
    plt.figure(figsize=(14, 5))
    plt.plot(time_axis, audio_sample)
    plt.title(f'Audio Waveform and Caption:\n"{caption_sample}"')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.show()
    break

