# Dual Lottery Ticket Hypothesis

This repository is for our ICLR'22 paper:
> Dual Lottery Ticket Hypothesis [arXiv](https://arxiv.org/abs/2203.04248) \
> [Yue Bai](https://yueb17.github.io/), [Huan Wang](http://huanwang.tech/), [Zhiqiang Tao](http://ztao.cc/), [Kunpeng Li](https://kunpengli1994.github.io/), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/)

This paper articulates a Dual Lottery Ticket Hypothesis (DLTH) as a dual format of original Lottery Ticket Hypothesis (LTH). Correspondingly, a simple regularization based sparse network training strategy, Random Sparse Network Transformation (RST), is proposed to validate DLTH and enhance sparse network training.

## Step 1: Set up environment
- python=3.6
- Install libraries by `pip install -r requirements.txt`.

## Setp 2: Running
```
# Pretraining, ResNet56, CIFAR100
CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet56 --dataset cifar100 --method L1 --stage_pr [0,0,0,0,0] --batch_size 128 --wd 0.0005 --lr_ft 0:0.1,100:0.01,150:0.001 --epochs 200 --project pretrain_resnet56_cifar100 --save_init_model
```

```
# RST One-shot, ResNet56, CIFAR100, sparsity ratio = 0.7
CUDA_VISIBLE_DEVICES=0 python main.py --arch resnet56 --dataset cifar100 --batch_size 128 --wd 0.0005 --lr_ft 0:0.1,100:0.01,150:0.001 --epochs 200 --wg weight --base_model_path Experiments/$YOUR SAVED PRETRAINED MODEL FOLDER NAME$/weights/checkpoint_just_finished_prune.pth --stage_pr [0,0.7,0.7,0.7,0] --method RST --project RST_rs56_cifar100_pr0.7
```

```
# RST Iter-5, ResNet56, CIFAR100, sparsity ratio = 0.7
CUDA_VISIBLE_DEVICES=0 python main.py --method RST_Iter --dataset cifar100 --arch resnet56 --wd 0.0005 --batch_size 128 --base_model_path Experiments/$YOUR SAVED PRETRAINED MODEL FOLDER NAME$/weights/checkpoint_just_finished_prune.pth --stage_pr [0,0.7,0.7,0.7,0] --lr_ft 0:0.1,100:0.01,150:0.001 --epochs 200 --num_cycles 5 --project RST_Iter5_rs56_cifar100_pr0.7 --wg weight --stabilize_reg_interval 10000 --update_reg_interval 1 --pick_pruned iter_rand --RST_Iter_ft 0
```



## Acknowledgments
We refer to the following repositories for our implementations: [Regularization-Pruning](https://github.com/MingSun-Tse/Regularization-Pruning), [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10). We appreciate their great works!

## Reference
Please cite this in your publication if our work helps your research. Should you have any questions, welcome to reach out to Yue Bai (bai.yue@northeastern.edu).

```
@inproceedings{bai2021dual,
  title={Dual Lottery Ticket Hypothesis},
  author={Bai, Yue and Wang, Huan and TAO, ZHIQIANG and Li, Kunpeng and Fu, Yun},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```


