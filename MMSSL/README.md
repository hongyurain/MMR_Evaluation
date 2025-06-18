# MMSSL: Multi-Modal Self-Supervised Learning for Recommendation

PyTorch implementation for WWW 2023 paper [Multi-Modal Self-Supervised Learning for Recommendation](https://arxiv.org/pdf/2302.10632.pdf).

<p align="center">
<img src="./MMSSL.png" alt="MMSSL" />
</p>

MMSSL is a new multimedia recommender system which integrates the generative modality-aware collaborative self-augmentation and the contrastive cross-modality dependency encoding. It achieves better performance than existing SOTA multi-model recommenders.


<h2>Dependencies </h2>

* Python >= 3.9.13
* [Pytorch](https://pytorch.org/) >= 1.13.0+cu116
* [dgl-cuda11.6](https://www.dgl.ai/) >= 0.9.1post1




<h2>Usage </h2>

Start training and inference as:

```
cd MMSSL
python ./main.py --dataset {DATASET}
```


<h1> Citing </h1>

If you find this work helpful to your research, please kindly consider citing our paper.


```
@inproceedings{wei2023multi,
  title={Multi-Modal Self-Supervised Learning for Recommendation},
  author={Wei, Wei and Huang, Chao and Xia, Lianghao and Zhang, Chuxu},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={790--800},
  year={2023}
}
```
<!-- or -->

<!-- @inproceedings{wei2023multi,
  title={Multi-Modal Self-Supervised Learning for Recommendation},
  author={Wei, Wei and Huang, Chao and Xia, Lianghao and Zhang, Chuxu},
  booktitle={Proceedings of the Web Conference (WWW)},
  year={2023}
}
 -->


## Acknowledgement

The structure of this code is largely based on [LATTICE](https://github.com/CRIPAC-DIG/LATTICE), [MICRO](https://github.com/CRIPAC-DIG/MICRO). Thank them for their work.

