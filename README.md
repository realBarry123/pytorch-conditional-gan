# 🚦 Conditional GAN

<img src="https://raw.githubusercontent.com/realBarry123/pytorch-conditional-gan/main/example.png" width="70%">

## Overview
Conditional GAN with auxiliary classifier, forced to generate MNIST digits. 

On top of the usual GAN formulation, a frozen classifier $C$ adds an additional term to the generator loss to encourage legible digits: 

$$\mathcal{L}_G ​= BCE(D(G(z)), 1) + \beta \cdot CE(C(G(z)), y)$$

### Files

* `train_c.py` trains a classic MNIST classifier. This is trained before the GAN. 
* `train.py` jointly trains a generator and discriminator.
* `model.py` defines model classes. 
* `data.py` defines utility functions for data manipulation. 
* Run `visualize.py` for a visualization of generation results.
