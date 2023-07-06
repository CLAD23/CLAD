# CLAD: Towards Robust Audio Deepfake Detection Against Manipulation Attacks with Contrastive Learning

This repository includes the code to reproduce our paper "CLAD: Towards Robust Audio Deepfake Detection Against Manipulation Attacks with Contrastive Learning".

Our paper is still under review. Current version contains the main part of the code of CLAD which contains the complete evaluation script of manipulation attacks.
You can evaluate the pretrained CLAD model to reproduce our results.

A more comprehensive version will be available soon.

## Get started

### Installation

`requirements.txt` must be installed for execution.

Note that we release our code in `ipynb` format, which will need a jupyter environment to run. You can change them into `py` format easily.

```
pip install -r requirements.txt
```

Our environment (for GPU training and evaluation)
  - GPU: 1 NVIDIA Tesla V100 32GB
    - About 27GB is required to train CLAD with AASIST as encoder using a batch size of 24
  - GPU Driver Version: 470.82.01 
  - CUDA Version 11.4

We also test our code on a Nvidia GeForce 4090 GPU server successfully.

### Data Preparation

You can download ASVspoof2019 dataset from https://datashare.ed.ac.uk/handle/10283/3336. We use the Logical part of it, so download LA.zip and unzip it.
After that, change the `database_path` in `config.conf` so that the script will be able to find it.

For the evaluation of environmental noise, you need to download [ESC-50 dataset](https://github.com/karolpiczak/ESC-50). After that, change the `noise_dataset_path` in `config.conf` file.

## Evalation

The training script of CLAD can be found in `Evaluation.ipynb`. Check the `config.conf` file and parameter part in the script before running it.

You can find CLAD model pretrained by us in `./pretrained_models/CLAD_pretrained_150_10.tar`.

Please make sure you have downloaded [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) and change the configurations as decribed in Data Preparation part.

### Preparation for the evaluation of AASIST

Please download config file of AASIST from https://github.com/clovaai/aasist/blob/main/config/AASIST.conf, and change the `aasist_config_path` to its path.

Then, download the pretrained AASIST model from https://github.com/clovaai/aasist/blob/main/models/weights/AASIST.pth, and change the `aasist_model_path` to its path.

### Preparation for the evaluation of RawNet2

Please download config file of AASIST from https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-RawNet2/model_config_RawNet.yaml, and change the `rawnet2_config_path` to its path.

Then, download the pretrained RawNet2 model from https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip, and change the `rawnet2_model_path` to its path.


## Acknowledgements
We would like to thank the authors and contributors of the following open source projects and datasets that we used in our work. We acknowledge their valuable efforts and generous sharing of their resources.

- [ASVspoof 2021 baseline repo](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2)
- [AASIST](https://github.com/clovaai/aasist/tree/main)
- [torch-time-stretch](https://github.com/KentoNishi/torch-time-stretch)
- [Moco](https://github.com/facebookresearch/moco)
- [ASVspoof 2019 dataset](https://www.asvspoof.org/index2019.html)
- [ESC-50 dataset](https://github.com/karolpiczak/ESC-50)



