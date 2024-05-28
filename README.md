[中文](README_CN.md) ｜ English

# Analytic Continual Learning

Official implementation of the following papers.

[1] Zhuang, Huiping, et al. "[ACIL: Analytic class-incremental learning with absolute memorization and privacy protection.](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4b74a42fc81fc7ee252f6bcb6e26c8be-Abstract-Conference.html)" Advances in Neural Information Processing Systems 35 (2022): 11602-11614.

[2] Zhuang, Huiping, et al. "[GKEAL: Gaussian kernel embedded analytic learning for few-shot class incremental task.](https://openaccess.thecvf.com/content/CVPR2023/html/Zhuang_GKEAL_Gaussian_Kernel_Embedded_Analytic_Learning_for_Few-Shot_Class_Incremental_CVPR_2023_paper.html)" Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[3] Zhuang, Huiping, et al. "[DS-AL: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning.](https://ojs.aaai.org/index.php/AAAI/article/view/29670)" Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 15. 2024.

[4] Zhuang, Huiping, et al. "[G-ACIL: Analytic Learning for Exemplar-Free Generalized Class Incremental Learning](https://arxiv.org/abs/2403.15706)" arXiv preprint arXiv:2403.15706 (2024).

![](figures/acc_cmp.jpg)

**Welcome to join our Tencent QQ group: [954528161](http://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=qaK4W8Jw6d--VWHlx7iUs93T2qMJT9k_&authKey=5e8hSXX8rALjM12iGwrZ9BmRBP9iUfCuRGNCaZ3%2Bx0msiRFVcSwu%2FuZpeKig1XQH&noverify=0&group_code=954528161). Chinese tutorial is available at [Bilibili](https://www.bilibili.com/video/BV1wq421A7YM/).**

# Dual Branch

We have a dual branch at "[Analytic Federated Learning.](https://github.com/ZHUANGHP/Analytic-federated-learning)" 

## Environment
We recommend using the [Anaconda](https://anaconda.org/) to install the development environment.

```bash
git clone --depth=1 git@github.com:ZHUANGHP/Analytic-continual-learning.git

cd Analytic-continual-learning
conda env create -f environment.yaml
conda activate AL

mkdir backbones
```

Download the pre-train weight at the [release page](https://github.com/ZHUANGHP/Analytic-continual-learning/releases) for quick start. We suggest you to extract the pre-train backbone (zip file) under the `backbones` folder.

For the macOS users and the CPU-only users, you need deleted the items related to CUDA in the `environment.yaml` file.

We highly recommend you to run our code in Linux. Windows and macOS users are also welcome to submit issues if they have problems running the code.

## Quick Start
Put the base training weights (provided at the [release page](https://github.com/ZHUANGHP/Analytic-continual-learning/releases)) at the `backbones` directory. Gradients are not used in continuous learning. **You can run our code even on CPUs.**

Here are some examples.

```bash
# ACIL (CIFAR-100, B50 25 phases)
python main.py ACIL --dataset CIFAR-100 --base-ratio 0.5 --phases 25 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone resnet32 \
    --gamma 0.1 --buffer-size 8192 \
    --cache-features --backbone-path ./backbones/resnet32_CIFAR-100_0.5_None
```
```bash
# G-ACIL (CIFAR-100, B50 25 phases)
python main.py G-ACIL --dataset CIFAR-100 --base-ratio 0.5 --phases 25 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone resnet32 \
    --gamma 0.1 --buffer-size 8192 \
    --cache-features --backbone-path ./backbones/resnet32_CIFAR-100_0.5_None
```
```bash
# GKEAL (CIFAR-100, B50 10 phases)
python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.5 --phases 10 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone resnet32 \
    --gamma 0.1 --sigma 10 --buffer-size 8192 \
    --cache-features --backbone-path ./backbones/resnet32_CIFAR-100_0.5_None
```
```bash
# DS-AL (CIFAR-100, B50 50 phases)
python main.py DS-AL --dataset CIFAR-100 --base-ratio 0.5 --phases 50 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone resnet32 \
    --gamma 0.1 --gamma-comp 0.1 --compensation-ratio 0.6 --buffer-size 8192 \
    --cache-features --backbone-path ./backbones/resnet32_CIFAR-100_0.5_None
```
```bash
# DS-AL (ImageNet-1k, B50 20 phases)
python main.py DS-AL --dataset ImageNet-1k --base-ratio 0.5 --phases 20 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone resnet18 \
    --gamma 0.1 --gamma-comp 0.1 --compensation-ratio 1.5 --buffer-size 16384 \
    --cache-features --backbone-path ./backbones/resnet18_ImageNet-1k_0.5_None
```

## Training From Scratch

```bash
# ACIL (CIFAR-100)
python main.py ACIL --dataset CIFAR-100 --base-ratio 0.5 --phases 25 \
    --data-root ~/dataset --batch-size 256 --num-workers 16 --backbone resnet32 \
    --learning-rate 0.5 --label-smoothing 0 --base-epochs 300 --weight-decay 5e-4 \
    --gamma 0.1 --buffer-size 8192 --cache-features --IL-batch-size 4096
```
```bash
# ACIL (ImageNet-1k)
python main.py ACIL --dataset ImageNet-1k --base-ratio 0.5 --phases 25 \
    --data-root ~/dataset --batch-size 256 --num-workers 16 --backbone resnet18 \
    --learning-rate 0.5 --label-smoothing 0.05 --base-epochs 300 --weight-decay 5e-5 \
    --gamma 0.1 --buffer-size 16384 --cache-features --IL-batch-size 4096
```

## Reproduction Details

### Difference Between the ACIL and the G-ACIL

The G-ACIL is a general version of the ACIL for the general CIL setting. For the tradition CIL setting, the G-ACIL is equivalent to the ACIL. Thus, we use the same implementation in this repository.

### Benchmarks (B50, 25 phases, with `TrivialAugmentWide`)

Metrics are shown in 95% confidence intervals ($\mu \pm 1.96\sigma$).

|   Dataset   | Method         | Backbone  | Buffer Size | Average Accuracy (%) | Last Phase Accuracy (%) |
| :---------: | :------------: | :-------: | :---------: | :------------------: | :---------------------: |
|  CIFAR-100  |  ACIL & G-ACIL | ResNet-32 |    8192     |   $71.047\pm0.252$   |    $63.384\pm0.330$     |
|  CIFAR-100  |  DS-AL         | ResNet-32 |    8192     |   $71.277\pm0.251$   |    $64.043\pm0.184$     |
|  CIFAR-100  |  GKEAL         | ResNet-32 |    8192     |   $70.371\pm0.168$   |    $62.301\pm0.191$     |
| ImageNet-1k |  ACIL & G-ACIL | ResNet-18 |    16384    |   $67.497\pm0.092$   |    $58.349\pm0.111$     |
| ImageNet-1k |  DS-AL         | ResNet-18 |    16384    |   $68.354\pm0.084$   |    $59.762\pm0.086$     |
| ImageNet-1k |  GKEAL         | ResNet-18 |    16384    |   $66.881\pm0.061$   |    $57.295\pm0.105$     |

![Top-1 Accuracy](figures/acc@1.svg)

### Hyper-Parameters (Analytic Continual Learning)
The backbones are frozen during the incremental learning process of our algorithm. You can use the `--cache-features` option to save the features output by the backbones to improve the efficiency of parameter adjustment.

1. **Buffer Size**

    For the ACIL, the buffer size means the *expansion size* of the random projection layer. For the GKEAL, the buffer size means the number of *center vectors* of the *Gaussian kernel embedding*. We summarize the "random projection" and the "Gaussian projection" into one concept "buffer" in the DS-AL.

    On most datasets, the performance of the algorithm first increases and then decreases as the buffer size increases. You can see further experiments on this hyperparameter in our papers. We recommend using a buffer size of 8192 on CIFAR-100 and 16384 or greater on ImageNet for optimal performance. A larger buffer size requires more memory.

2. **$\gamma$ (Coefficient of the Regularization Term)**

    For the dataset used in the papers, $\gamma$ is insensitive within a interval. However, a $\gamma$ that is too small may cause numerical stability problems in matrix inversion, and a $\gamma$ that is too large may cause under-fitting of the classifier. On both CIFAR-100 and ImageNet-1k, $\gamma$ is 0.1. When you migrate our algorithm to other datasets, we still recommend that you do some experiments to check whether $\gamma$ is appropriate.

3. **$\beta$ and $\sigma$ (GKEAL Only)**

    In the GKEAL, the width-adjusting parameter $\beta$ controls the width of the Gaussian kernels. There is a comfortable range for $\sigma$ at around $[5, 15]$ for CIFAR-100 and ImageNet-1k that gives good results, where $\beta = \frac{1}{2\sigma^2}$.

4. **Compensation Ratio $\mathcal{C}$ (DS-AL Only)**

    We recommend using the grid search to find the best compensation ratio in the interval $[0, 2]$. The best value is 0.6 for the CIFAR-100, while the best value for the ImageNet-1k is 1.5.

Further analysis on hyper-parameters are shown in our papers.

### Hyper-Parameters (Base Training)
In the base training process, the backbones reaches over 80% top-1 accuracy on the first half of CIFAR-100 (ResNet-32) and ImageNet-1k (ResNet-18). Important hyper-parameters are listed below.

1. **Learning Rate**

    In this implementation, we use a cosine scheduler instead of choosing the same piece-wise smooth scheduler as in the papers to reduce the number of hyper-parameters. We recommend using a learning rate of 0.5 (when the batch size is 256) on CIFAR-100 and ImageNet-1k to obtain better convergence. The number of epochs we use for provided backbones is 300.

2. **Label Smoothing and Weight Decay**

    Properly setting label smoothing and weight decay can help prevent over-fitting of the backbone. In CIFAR-100, label smoothing is not significantly helpful; while in ImageNet-1k, we empirically selected 0.05. For CIFAR-100, we choose a weight decay of 5e-4, while in ImageNet-1k, this value is 5e-5.

3. **Image Augmentation**

    Using image augmentation to obtain a more generalizable backbone in the base training dataset can significantly improve performance. No image augmentation is used in the experiments of our papers. But in this implementation, data augmentation is enabled on by default. **So using this implementation will achieve higher performance than reported in the papers (about 2%~5%)**.

    Note that we do not use any data augmentation during the re-alignment and the continual learning processes because each sample will be learned only once.

# Cite Our Papers

```bib
@InProceedings{Zhuang_ACIL_NeurIPS2022,
    author    = {ZHUANG, HUIPING and Weng, Zhenyu and Wei, Hongxin and XIE, RENCHUNZI and Toh, Kar-Ann and Lin, Zhiping},
    title     = {{ACIL}: Analytic Class-Incremental Learning with Absolute Memorization and Privacy Protection},
    booktitle = {Advances in Neural Information Processing Systems},
    editor    = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
    pages     = {11602--11614},
    publisher = {Curran Associates, Inc.},
    url       = {https://proceedings.neurips.cc/paper_files/paper/2022/file/4b74a42fc81fc7ee252f6bcb6e26c8be-Paper-Conference.pdf},
    volume    = {35},
    year      = {2022}
}

@InProceedings{Zhuang_GKEAL_CVPR2023,
    author    = {Zhuang, Huiping and Weng, Zhenyu and He, Run and Lin, Zhiping and Zeng, Ziqian},
    title     = {{GKEAL}: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {7746-7755},
    url       = {https://openaccess.thecvf.com/content/CVPR2023/papers/Zhuang_GKEAL_Gaussian_Kernel_Embedded_Analytic_Learning_for_Few-Shot_Class_Incremental_CVPR_2023_paper.pdf}
}

@article{Zhuang_DSAL_AAAI2024,
    title     = {{DS-AL}: A Dual-Stream Analytic Learning for Exemplar-Free Class-Incremental Learning},
    author    = {Zhuang, Huiping and He, Run and Tong, Kai and Zeng, Ziqian and Chen, Cen and Lin, Zhiping},
    journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
    volume    = {38},
    number    = {15},
    pages     = {17237-17244},
    year      = {2024},
    month     = {Mar.},
    DOI       = {10.1609/aaai.v38i15.29670},
    url       = {https://ojs.aaai.org/index.php/AAAI/article/view/29670},
}

@misc{Zhuang_GACIL_arXiv2024,
    title         = {{G-ACIL}: Analytic Learning for Exemplar-Free Generalized Class Incremental Learning}, 
    author        = {Huiping Zhuang and Yizhu Chen and Di Fang and Run He and Kai Tong and Hongxin Wei and Ziqian Zeng and Cen Chen},
    year          = {2024},
    eprint        = {2403.15706},
    archivePrefix = {arXiv},
    primaryClass  = {cs.LG}
}
```
