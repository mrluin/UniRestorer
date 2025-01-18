## Preparing Datasets

### Deraining

|       | All-in-One comparison | Single-task comparison                        |
|-------|-----------------------|-----------------------------------------------|
| Train | [Rain100L](https://github.com/csdwren/PReNet)          | [Rain200L](https://github.com/cschenxiang/DRSformer), [Rain200H](https://github.com/cschenxiang/DRSformer), [DID](https://github.com/cschenxiang/DRSformer), [DDN](https://github.com/cschenxiang/DRSformer)  |
| Test  | [Rain100L](https://github.com/csdwren/PReNet)          | [Rain200L](https://github.com/cschenxiang/DRSformer), [Rain200H](https://github.com/cschenxiang/DRSformer), [DID](https://github.com/cschenxiang/DRSformer), [DDN](https://github.com/cschenxiang/DRSformer)  |


### Dehazing

|       | All-in-One comparison | Single-task comparison |
|-------|-----------------------|------------------------|
| Train | [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2?authuser=0)            | [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2?authuser=0)             |
| Test  | [SOTS-outdoor](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=0)      | [SOTS-outdoor](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=0)       |

**Note that**, in RESIDE dataset, there has some overlap between training data and testing data, before training we exclude these overlapped sampels in training set.

### Desnowing

|       | All-in-One comparison | Single-task comparison |
|-------|-----------------------|------------------------|
| Train | [Snow100K](https://sites.google.com/view/yunfuliu/desnownet)          | [Snow100K](https://sites.google.com/view/yunfuliu/desnownet)           |
| Test  | [Snow100K-test](https://sites.google.com/view/yunfuliu/desnownet)     | [Snow100K-test](https://sites.google.com/view/yunfuliu/desnownet)      |

**Note that**, we evaluate the performance on the whole Snow100K-test. Previous works may evaluate their performance on a susbet of Snow100K-test, but they did not clarify how to construct testing data.


### Motion-Deblurring

|       | All-in-One comparison | Single-task comparison |
|-------|-----------------------|------------------------|
| Train | [GoPro](https://seungjunnah.github.io/Datasets/gopro.html)             | [GoPro](https://seungjunnah.github.io/Datasets/gopro.html)              |
| Test  | [GoPro](https://seungjunnah.github.io/Datasets/gopro.html)             | [GoPro](https://seungjunnah.github.io/Datasets/gopro.html)              |

### Gaussian Color Image Denoising


|       | All-in-One comparison                     | Single-task comparison                     |
|-------|-------------------------------------------|--------------------------------------------|
| Train | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar), [BSD](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar), [BSD](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)  |
| Test  | [BSD68](https://github.com/cszn/DnCNN/tree/master/testsets)                                 | [Urban100](https://github.com/jbhuang0604/SelfExSR)                               |

### Compression Artifacts Removal


|       | All-in-One comparison                                                                                                                                                                                                                                                                                        | Single-task comparison                                                                                                                                                                                                                                                                                       |
|-------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Train | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar), [BSD](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar), [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar), [BSD](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) |
| Test  | [LIVE1](https://github.com/cszn/DnCNN/tree/master/testsets)                                                                                                                                                                                                                                                  | [LIVE1](https://github.com/cszn/DnCNN/tree/master/testsets)                                                                                                                                                                                                                                                  |

### Lowlight Enhancement


|       | All-in-One comparison                                             | Single-task comparison                                         |
|-------|-------------------------------------------------------------------|----------------------------------------------------------------|
| Train | [LOLv1](https://github.com/caiyuanhao1998/Retinexformer)          | [LOLv1](https://github.com/caiyuanhao1998/Retinexformer)       |
| Test  | [LOLv1](https://github.com/caiyuanhao1998/Retinexformer)          | [LOLv1](https://github.com/caiyuanhao1998/Retinexformer)       |

We follow dataset usage as [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer).

### Multiple Mixed Degradation

|       | All-in-One comparison                                                                                           | 
|-------|-----------------------------------------------------------------------------------------------------------------|
| Train | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/), [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) | 
| Test  | [DIV2K-val](https://data.vision.ee.ethz.ch/cvl/DIV2K/)                                                          | 

The way of degradation follows [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [BSRGAN](https://github.com/cszn/BSRGAN), [MiOIR](https://github.com/Xiangtaokong/MiOIR), and [OneRestore](https://github.com/gy65896/OneRestore).
