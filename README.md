# CLAD: Robust Audio Deepfake Detection Against Manipulation Attacks with Contrastive Learning

This repository includes the code to reproduce our paper "CLAD: Robust Audio Deepfake Detection Against Manipulation Attacks with Contrastive Learning".

Our paper is still under review. Code here contains the complete evaluation script of manipulation attacks. You can evaluate the pretrained CLAD model and other baselines to reproduce our results.

## Get started

### Installation

`requirements.txt` must be installed for execution.

Note that we release our code in `ipynb` format, which will need a Jupyter environment to run. You can change them into `py` format easily.

```
pip install -r requirements.txt
```

Our environment (for GPU training and evaluation)
  - GPU: 1 NVIDIA Tesla V100 32GB
    - About 24GB is required to train CLAD with AASIST as encoder using a batch size of 24
  - GPU Driver Version: 470.82.01 
  - CUDA Version 11.4

We also test our code on a Nvidia GeForce 4090 GPU server successfully.

### Data Preparation

You can download ASVspoof2019 dataset from https://datashare.ed.ac.uk/handle/10283/3336. We use the Logical part of it, so download LA.zip and unzip it.
After that, change the `database_path` in `config.conf` so that the script will be able to find it.

For the evaluation of environmental noise, you need to download [ESC-50 dataset](https://github.com/karolpiczak/ESC-50). After that, change the `noise_dataset_path` in `config.conf` file.

<!-- ## Training -->

<!-- The training script of CLAD can be found in `Training.ipynb`. Check the `config.conf` file and parameter part in the script before running it. -->

## Evaluation

The evaluation script of CLAD can be found in `Evaluation.ipynb`. Check the `config.conf` file and parameter part in the script before running it.

We have included a pretrained CLAD model, which you can find in the directory `./pretrained_models/CLAD_150_10_2310.pth.tar`.

Please make sure you have downloaded [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) and change the configurations as described in Data Preparation part.

Please be aware that when employing white noise injection, your results may slightly differ from ours. This is due to the fact that each time you execute the code, the generated white noise will be distinct. However, for other data manipulations, you should expect to obtain consistent results.

### Preparation for the evaluation of AASIST

Please download config file of AASIST from https://github.com/clovaai/aasist/blob/main/config/AASIST.conf, and change the `aasist_config_path` to its path.

Then, download the pretrained AASIST model from https://github.com/clovaai/aasist/blob/main/models/weights/AASIST.pth, and change the `aasist_model_path` to its path.

### Preparation for the evaluation of RawNet2

Please download config file of RawNet2 from https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-RawNet2/model_config_RawNet.yaml, and change the `rawnet2_config_path` to its path.

Then, download the pretrained RawNet2 model from https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip, and change the `rawnet2_model_path` to its path.

### Preparation for the evaluation of Res-TSSDNet

Res-TSSDNet does not use a config file, and you just need to download the [pretrained model released by authors](https://github.com/ghua-ac/end-to-end-synthetic-speech-detection/blob/main/pretrained/Res_TSSDNet_time_frame_61_ASVspoof2019_LA_Loss_0.0017_dEER_0.74%25_eEER_1.64%25.pth) and change the `res_tssdnet_model_path` in the config to its path.

### Preparation for the Evaluation of SAMO

To evaluate the SAMO model, we have incorporated its code directly into our repository. The only modification made is to the import lines to ensure it works correctly. Follow these steps to prepare for the SAMO evaluation:

1. Download the `aasist` directory from the [SAMO repository](https://github.com/sivannavis/samo/tree/main/samo/aasist) and place it in the root of this repository. (This is required for loading the SAMO model released by the authors.)

2. Download the pretrained SAMO model from the [official release](https://github.com/sivannavis/samo/blob/main/models/samo.pt), and update the `samo_model_path` variable with the path to the downloaded model file.

## Acknowledgements
We would like to thank the authors and contributors of the following open source projects and datasets that we used in our work. We acknowledge their valuable efforts and generous sharing of their resources.

- [ASVspoof 2021 baseline repo](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2)
- [AASIST](https://github.com/clovaai/aasist/tree/main)
- [Res-TSSDNet implementation](https://github.com/ghua-ac/end-to-end-synthetic-speech-detection)
- [SAMO](https://github.com/sivannavis/samo)
- [torch-time-stretch](https://github.com/KentoNishi/torch-time-stretch)
- [MoCo](https://github.com/facebookresearch/moco)
- [ASVspoof 2019 dataset](https://www.asvspoof.org/index2019.html)
- [ESC-50 dataset](https://github.com/karolpiczak/ESC-50)



