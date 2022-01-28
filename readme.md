# Understanding Why Generalized Reweighting Does Not Improve Over ERM
Runtian Zhai, Chen Dan, Zico Kolter, Pradeep Ravikumar  
Carnegie Mellon University

## Table of Contents
- [Quick Start](#quick-start)
- [Introduction](#introduction)
- [Training](#training)
- [Example Commands](#example-commands)
- [Contact](#contact)

## Quick Start
Before training, please download the datasets first. You can download Waterbirds at this [link](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz), and CelebA [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

If you want to exactly reproduce our experimental results, please create a virtual environment with Anaconda using:
```shell
conda env create --file env.yml
```


To train a model on Waterbirds, run the following command:
```shell
python waterbirds_train.py --data_root /path/to/dataset --alg [ALG] --wd [WD] --test_train --seed [SEED]
```
where `[ALG]` is `erm`, `iw` or `gdro`, `[WD]` is the weight decay level and `[SEED]` is the random seed.

To train a model on CelebA, run the following command:
```shell
python celeba_train.py --data_root /path/to/dataset --alg [ALG] --wd [WD] --test_train --seed [SEED]
```

## Introduction

In this work, we theoretically prove the pessimistic result that all reweighting algorithms overfit, and if regularization is applied, it must be large enough to prevent the model from achieving nearly perfect training performance in order to avoid overfitting.

This repository contains codes for experiments to empirically validate our theoretical results. Particularly, we conduct the experiments on two datasets: Waterbirds and CelebA.

The first part of our experiments (synthetic dataset) can be found at https://colab.research.google.com/drive/1Yt2MsAvOhZ0Rf0pFK1AqSaO7WaVpFX5p?usp=sharing.


## Training

On each of the dataset, we use a ResNet 18 as the model and optimize it with momentum SGD. Our codes provide command-line options for learning rate (`--lr`), weight decay level (`--wd`) and multi-level learning rate decay scheduler (`--scheduler`), so it is very simple to train a model under different settings for optimization.

For instance, to train a model on CelebA with Group DRO, learning rate `0.001`, weight decay `0.01` for 300 epochs with the learning rate decayed at Epochs 200 and 250, simply run:
```shell
python celeba_train.py --data_root /path/to/dataset --alg gdro --lr 0.001 --wd 0.01 --epochs 300 --scheduler 200,250 --test_train --seed [SEED]
```

In our experiments, we use the following fixed set of random seeds: `2002, 2022, 2042, 2062, 2082`.

## Example Commands
These are some of the commands we used in our experiments:
```shell
CUBLAS_WORKSPACE_CONFIG=:4096:8 python waterbirds_train.py --data_root data --alg erm --batch_size 128 --wd 0 --lr 0.0001 --save_file wb_erm_wd0_2022.mat --seed 2022 --epochs 500 --test_train

CUBLAS_WORKSPACE_CONFIG=:4096:8 python celeba_train.py --data_root data --alg iw --batch_size 400 --wd 0.1 --lr 0.001 --save_file celeba_iw_wd01_2062.mat --seed 2062 --epochs 250 --test_train

CUBLAS_WORKSPACE_CONFIG=:4096:8 python celeba_train.py --data_root data --alg gdro --batch_size 400 --wd 0.01 --lr 0.001 --save_file celeba_gdro_wd001_2002.mat --seed 2002 --epochs 250 --test_train
```
All our results can be exactly reproduced by running these commands (with proper arguments) in the same environment.

## Contact
To contact us, please email to the following address: `Runtian Zhai <rzhai@cmu.edu>`