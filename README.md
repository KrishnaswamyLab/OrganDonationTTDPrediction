# Deep Learning for Organ Donation Time-to-Death Prediction

Official implementation of **"Deep learning unlocks the true potential of organ donation after circulatory death with accurate prediction of time-to-death"** published at *Nature Scientific Reports* (https://www.nature.com/articles/s41598-025-95079-7)

## Overview
This repository contains the implementation of deep learning models for time-to-death prediction after circulatory death (DCD). Our model (ODE–RNN) overcomes the computational challenges of irregular, sparse clinical time-series data and effectively integrates clinical history. It outperforms existing approaches on external validations, thereby facilitating organ donation preparation. 

## Models
- **GRU-dt**: Gated Recurrent Unit with delta time
- **GRU-D**: Gated Recurrent Unit with Decay
- **ODE-RNN**: Neural Ordinary Differential Equation Recurrent Neural Network

## RNN Variants
We experiment with different recurrent architectures:
- **RNN**: Simple Recurrent Neural Network
- **LSTM**: Long Short-Term Memory
- **GRU**: Gated Recurrent Unit

## Citation
If you find this work useful, please cite our paper:
```bibtex
@article{sun2025deep,
  title={Deep learning unlocks the true potential of organ donation after circulatory 
  death with accurate 
  prediction of time-to-death},
  author={Sun, Xingzhi and De Brouwer, Edward and Liu, Chen and Krishnaswamy, Smita and 
  Batra, Ramesh},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={13565},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```