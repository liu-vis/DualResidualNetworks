# Dual Residual Networks  
By Xing Liu<sup>1</sup>, [Suganuma Masanori](https://scholar.google.co.jp/citations?user=NpWGfwgAAAAJ&hl=ja)<sup>1,2</sup>, [Zhun Sun](https://scholar.google.co.jp/citations?user=Y-3iZ9EAAAAJ&hl=en)<sup>2</sup>, [Takayuki Okatani](https://scholar.google.com/citations?user=gn780jcAAAAJ&hl=en)<sup>1,2</sup>

Tohoku University<sup>1</sup>, RIKEN Center for AIP<sup>2</sup>

[link to the paper](https://arxiv.org/pdf/1903.08817.pdf)

## Table of Contents
1) Abstract

2) Citation

3) Numerical Results

4) Visual Results

5) Datasets

6) Models

7) Train

8) Test

## Abstract
In this paper, we study design of deep neural networks for tasks of image restoration. We propose a novel style of residual connections dubbed “dual residual connection”, which exploits the potential of paired operations, e.g., upand down-sampling or convolution with large- and smallsize kernels. We design a modular block implementing this connection style; it is equipped with two containers to which arbitrary paired operations are inserted. Adopting the “unraveled” view of the residual networks proposed by Veit et al., we point out that a stack of the proposed modular blocks allows the first operation in a block interact with the second operation in any subsequent blocks. Specifying the two operations in each of the stacked blocks, we build a complete network for each individual task of image restoration. We experimentally evaluate the proposed approach on five image restoration tasks using nine datasets. The results show that the proposed networks with properly chosen paired operations outperform previous methods on almost all of the tasks and datasets.


## Citation
```
@inproceedings{DuRN19,
title={Dual Residual Networks Leveraging the Potential of Paired Operations for Image Restoration},
author={Liu, Xing and Suganuma, Masanori and Sun, Zhun and Okatani, Takayuki},
booktitle={arXiv preprint arXiv:1903.08817},
year={2019},
}
```

## Numerical results
Please find them in the <code>test/results_confirmed.txt</code> file.

## Visual results
### Noise removal
![alt text]()
