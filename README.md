# UniRestorer
PyTorch implementation of [**UniRestorer: Universal Image Restoration via Adaptively Estimating Image Degradation at Proper Granularity**](https://github.com/mrluin/UniRestorer)


[![ArXiv](https://img.shields.io/badge/UniRestorer-ArXiv-red.svg)](https://arxiv.org/abs/2412.20157)
[![Paper](https://img.shields.io/badge/UniRestorer-Paper-blue.svg)](./assets/_arxiv_UniRestorer__Universal_Image_Restoration_via_Adaptively_Estimating_Image_Degradation_at_Proper_Granularity.pdf)


<hr />

> **Abstract:** *Recently, considerable progress has been made in all-in-one image restoration. Generally, existing methods can be degradation-agnostic or degradation-aware.
However, the former are limited in leveraging degradation-specific restoration, and the latter suffer from the inevitable error in degradation estimation.
Consequently, the performance of existing methods has a large gap compared to specific single-task models.
In this work, we make a step forward in this topic, and present our UniRestorer with improved restoration performance.
Specifically, we perform hierarchical clustering on degradation space, and train a multi-granularity mixture-of-experts (MoE) restoration model.
Then, UniRestorer adopts both degradation and granularity estimation to adaptively select an appropriate expert for image restoration.
In contrast to existing degradation-agnostic and -aware methods, UniRestorer can leverage degradation estimation to benefit degradation-specific restoration, and use granularity estimation to make the model robust to degradation estimation error.
Experimental results show that our UniRestorer outperforms state-of-the-art all-in-one methods by a large margin, and is promising in closing the performance gap to specific single-task models.* 
<hr />

## 1. TODO
-  [ ] Paper and supplement release on arXiv.
-  [ ] Inference code and pre-trained models release.
-  [ ] Datasets, training code release.

## 2. Method

<p align="center"><img src="./assets/framework.png" width="95%"></p>

<p>  Our method has three steps, constructing multi-granularity degradation set, train multi-granularity MoE restoration model, train degradation and granularity estimation-based routing. </p>


## 3. Comparisons
<details>
<summary><strong>All-in-One Image Restoration (single-degradation)</strong> (click to expand) </summary>

<img src = "./assets/all_in_one_singledeg.png"> 
</details>
<details>

<summary><strong>All-in-One Image Restoration (mixed-degradation)</strong> (click to expand) </summary>
<img src = "./assets/all_in_one_mixeddeg.png"> 
</details>

<details>

<summary><strong>Compare to Single-task Models</strong> (click to expand) </summary>
<img src = "./assets/compare_to_singletask_model.png"> 
</details>

## Contact & Acknowledgement

If you have any questions, please contact [jblincs1996@gmail.com](jblincs1996@gmail.com).

We thank to the following image restoration works for their awesome backbones and code repos:

- [Restormer](https://github.com/swz30/Restormer)
- [RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer)
- [DehazeFormer](https://github.com/IDKiro/DehazeFormer)
- [MiOIR](https://github.com/Xiangtaokong/MiOIR)
- [OneRestore](https://github.com/gy65896/OneRestore)


## Citation

Comming soon.
