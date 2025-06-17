# LightGT: A Light Graph Transformer for Multimedia Recommendation
This is our Pytorch implementation for the [LightGT](https://dl.acm.org/doi/10.1145/3539618.3591716):  
> Yinwei Wei, Wenqi Liu, Fan Liu, Xiang Wang, Liqiang Nie and Tat-Seng Chua (2023). LightGT: A Light Graph Transformer for Multimedia Recommendation. In ACM SIGIR`23, Taipei, July. 23-27, 2023

<img src="https://github.com/Liuwq-bit/LightGT/blob/master/image/figure1.png" width="50%" height="50%"><img src="https://github.com/Liuwq-bit/LightGT/blob/master/image/figure2.png" width="50%" height="50%">

## Environment Requirement
The code has been tested running under Python 3.8.15. The required packages are as follows:
- Pytorch == 1.7.0
- numpy == 1.23.4



-`train.npy`
   Train file. Each line is a user with her/his positive interactions with items: (userID and micro-video ID)  
-`val.npy`
   Validation file. Each line is a user several positive interactions with items: (userID and micro-video ID)  
-`test.npy`
   Test file. Each line is a user with several positive interactions with items: (userID and micro-video ID)  

## Citation
If you want to use our codes and datasets in your research, please cite:

``` 
@inproceedings{wei2023lightgt,
  title      = {Lightgt: A light graph transformer for multimedia recommendation},
  author     = {Wei, Yinwei and
                Liu, Wenqi and
                Liu, Fan and
                Wang, Xiang and
                Nie, Liqiang and
                Chua, Tat-Seng},
  booktitle  = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages      = {1508--1517},
  year       = {2023}
}
```
