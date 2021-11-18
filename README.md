# STN-CoordConv
Pytorch implementation of spatial transformer networks (STN) and CoordConv for ConvLayers with some experiments on toy datasets.


#### Implemented Methods/Papers
* [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)
* [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
* [STN-OCR: A single Neural Network for Text Detection and Text Recognition](https://arxiv.org/abs/1707.08831)


#### Requirements
The experiments were performed using Python 3.8.5 with the following Python packages:
- [numpy](http://www.numpy.org/) == 1.18.5
- [torch](https://pytorch.org/) == 1.5.1
- [torchvision](https://pypi.org/project/torchvision/) = 0.6.1
- [matplotlib](https://pypi.org/project/matplotlib/) == 3.3.3



### Usage
To play with my implementation, you can simply put the following command into your terminal after adjusting the necessary parameters:
```
python3 main.py [--seed SEED] [--use_cuda USE_CUDA]
                [--batch_size BATCH_SIZE] [--lr LR]
                [--num_workers NUM_WORKERS] 
                [--num_epochs NUM_EPOCHS] 
                [--optimizer OPTIMIZER] [--beta1 BETA1] [--beta2 BETA2]
```


#### Acknowledgements
1. implementation of STN by [Ankit](https://github.com/aicaffeinelife/Pytorch-STN).
2. implementation of CoordConv by [Chao Wen](https://github.com/walsvid/CoordConv).
