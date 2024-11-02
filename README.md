## Intro

This reporsitory supports the use of pytorch's `Dataloader` with clotho dataset.

## Download dataset.

 You should download the dataset in [here](https://zenodo.org/record/3490684)

## Preprocessing

We write a demo, and you can customize `preprocess.py` to get your own features.

```shell
python preprocess.py
```

## Hyperparamters

You can setup hyperparameters in `config/config.yaml` and use them in code by `import hparam as hp`.

## Testing dataloader

You can test the dataloader by running `main.py`

```shell
python main.py
```
