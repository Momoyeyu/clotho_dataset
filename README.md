## Intro

This repository supports the use of pytorch's `Dataloader` with clotho dataset.

## Download dataset.

You can download the dataset in [here](https://zenodo.org/record/3490684).
Scroll down to find `clotho_audio_development.7z` and `clotho_audio_evaluation.7z` and download them.
After downloading the above 7z files, unzip them into the `data/` directory.
You can also put them wherever you like, but you should update the `config.yaml` in `config/` directory.

## Preprocessing

We write a demo to preprocess the `wav` data into numpy vectors.
You can test the demo by running the following instruction.

```shell    
python preprocess.py
```

You can customize `preprocess.py` to get your own features.

## Hyperparameters

You can set up hyperparameters in `config/config.yaml` and use them in code by `import hparam as hp`.

## Testing dataloader

You can test the dataloader by running `main.py`

```shell
python main.py
```
