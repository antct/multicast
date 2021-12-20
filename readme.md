# MULTICAST

Code base for "Empower Distantly Supervised Relation Extraction with Collaborative Adversarial Training".

## Env

Refer to repo [OpenNRE](https://github.com/thunlp/OpenNRE), use `pip` to install required libraries → [requirements.txt](https://github.com/thunlp/OpenNRE/blob/master/requirements.txt).

## Run

1. Run script `download.py` to download dataset `nyt` and embedding `glove`.
```bash
python download.py
```

2. Run script `train.sh` to train and evaluate the DSRE models.
```bash
bash train.sh SEED
```

## Cite

```text
@inproceedings{chen2021empower,
  title={Empower Distantly Supervised Relation Extraction with Collaborative Adversarial Training},
  author={Chen, Tao and Shi, Haochen and Liu, Liyuan and Tang, Siliang and Shao, Jian and Chen, Zhigang and Zhuang, Yueting},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={14},
  pages={12675--12682},
  year={2021}
}
```



