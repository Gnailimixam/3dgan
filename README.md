# tf-3dgan

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/meetshah1995/tf-3dgan/blob/master/LICENSE)
[![arXiv Tag](https://img.shields.io/badge/arXiv-1610.07584-brightgreen.svg)](https://arxiv.org/abs/1610.07584)

## Tensorflow implementation of 3D Generative Adversarial Network.

This is a tensorflow implementation of the paper "Learning a Probabilistic Latent Space of Object Shapes 
via 3D Generative-Adversarial Modeling".

![](http://3dgan.csail.mit.edu/images/model.jpg)


### Requirements

* tensorflow=1.0
* scipy
* scikit-image
* trimesh
* stl (optional)
* matplotlib


#### One-line installation

    ```
    pip install scipy scikit-image stl trimesh matplotlib
    ```

### Data

* Download the training data from the 3D Shapenet [website](http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip)
* Extract the zip and modify the path appropriately in `dataIO.py`

### Usage

To train the model:

```
python 3dgan_mit_biasfree.py 0 <path_to_model_checkpoint>
```

To generate chairs:

```
python 3dgan_mit_biasfree.py 1 <path_to_trained_model>
```

Some sample generated chairs

|            |              |            |          |          |
|------------|--------------|------------|----------|----------|
|![](http://homepages.iitb.ac.in/~meetshah1995/files/2_.png) | ![](http://homepages.iitb.ac.in/~meetshah1995/files/4_.png) |  ![](http://homepages.iitb.ac.in/~meetshah1995/files/1_.png) |  ![](http://homepages.iitb.ac.in/~meetshah1995/files/3_.png) |  ![](http://homepages.iitb.ac.in/~meetshah1995/files/5_.png) |


### Source code files

| File      | Description                                                                   |
|-----------|-------------------------------------------------------------------------------|
|3dgan_mit_biasfree.py      | 3dgan as mentioned in the paper, with same hyperparameters. 
|3dgan.py                   | baseline 3dgan with fully connected layer at end of discriminator.
|3dgan_mit.py               | 3dgan as mentioned in the paper with bias in convolutional layers.
|3dgan_autoencoder.py       | 3dgan with support for autoencoder based pre-training.
|3dgan_feature_matching.py  | 3dgan with additional loss of feature matching of last layers. 
|dataIO.py                  | data input output and plotting utilities.
|utils.py                   | tensorflow utils like leaky_relu and batch_norm layer.


### Todo

* Host the trained models
* Add argparser based interface
* Train for longer number of epochs to improve quality of generated chairs. 



Main Code forked from Meetshah

