# NeighConsensus
NCNet Re-implementation on **Pytorch 1.0 + Python3.6** (Original implementation : https://github.com/ignacio-rocco/ncnet)
For more information check out their project [website](https://www.di.ens.fr/willow/research/ncnet/) and their paper on [arXiv](https://arxiv.org/abs/1810.10510).

## Table of Content
* [Visual Results](#visual-results)
* [Installation](#installation)
* [Functions Quick Search](https://github.com/XiSHEN0220/NeighConsensus/blob/master/model/README.md)
* [Train](#train)
* [Evaluation](#evaluation)




### Visual Results

More visual results can be found [here](http://imagine.enpc.fr/~shenx/visualRes/Match.html)

| TOP 5 % Matches | TOP 10 % Matches | TOP 20 % Matches | Ground Truth | Key Points Match |
| --- | --- | --- | --- | --- |
|![](https://github.com/XiSHEN0220/NeighConsensus/blob/master/img/Top5_14.jpg) | ![](https://github.com/XiSHEN0220/NeighConsensus/blob/master/img/Top10_14.jpg) | ![](https://github.com/XiSHEN0220/NeighConsensus/blob/master/img/Top20_14.jpg) | ![](https://github.com/XiSHEN0220/NeighConsensus/blob/master/img/GT14.jpg) | ![](https://github.com/XiSHEN0220/NeighConsensus/blob/master/img/KeyPointMatch14.jpg) |
|![](https://github.com/XiSHEN0220/NeighConsensus/blob/master/img/Top5_53.jpg) | ![](https://github.com/XiSHEN0220/NeighConsensus/blob/master/img/Top10_53.jpg) | ![](https://github.com/XiSHEN0220/NeighConsensus/blob/master/img/Top20_53.jpg) | ![](https://github.com/XiSHEN0220/NeighConsensus/blob/master/img/GT53.jpg) | ![](https://github.com/XiSHEN0220/NeighConsensus/blob/master/img/KeyPointMatch53.jpg) |
|![](https://github.com/XiSHEN0220/NeighConsensus/blob/master/img/Top5_74.jpg) | ![](https://github.com/XiSHEN0220/NeighConsensus/blob/master/img/Top10_74.jpg) | ![](https://github.com/XiSHEN0220/NeighConsensus/blob/master/img/Top20_74.jpg) | ![](https://github.com/XiSHEN0220/NeighConsensus/blob/master/img/GT74.jpg) | ![](https://github.com/XiSHEN0220/NeighConsensus/blob/master/img/KeyPointMatch74.jpg) |


## Installation

To download Pre-trained feature extractor (ResNet 18 ): 

``` Bash
cd model/FeatureExtractor
bash download.sh
```

To download PF Pascal Dataset : 

``` Bash
cd data/pf-pascal/
bash download.sh
```


### Functions Quick Search

For important functions, we provide a quick search [here](https://github.com/XiSHEN0220/NeighConsensus/blob/master/model/README.md)

### Train 

To train on PF-Pascal : 
``` Bash
bash demo_train.sh
``` 

### Evaluation

To evaluate on PF-Pascal : 
``` Bash
bash demo_eval.sh
```



