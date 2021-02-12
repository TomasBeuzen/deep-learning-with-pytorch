# Deep Learning with PyTorch

**By [Tomas Beuzen](https://www.tomasbeuzen.com/) 🚀**

Welcome to Deep Learning with PyTorch! With this website I aim to provide an introduction to optimization, neural networks and deep learning using PyTorch. We will progressively build up our deep learning knowledge, covering topics such as optimization algorithms like gradient descent, fully connected neural networks for regression and classification tasks, convolutional neural networks for image classification, transfer learning, and even generative adversarial networks (GANs) for synthetic image generation!

<p align="center">
  <img src="docs/logo.png" width="260">
</p>

>If you're interested in learning more about Python programming, you can check out my other online material [**Python Programming for Data Science**](https://www.tomasbeuzen.com/python-programming-for-data-science/). Or, if you'd like to learn more about making your very own Python packages, check out my and [Tiffany Timber's](https://www.tiffanytimbers.com/) open-source book [**Python Packages**](https://py-pkgs.org/).

>The content of this site is adapted from material I used to teach the 2020/2021 offering of the course "DSCI 572 Supervised Learning II" for the University of British Columbia's Master of Data Science Program. That material has built upon previous course material developed by [Mike Gelbart](https://www.mikegelbart.com/). A big thank you also goes to [Aaron Berk](https://aaronberk.ca/) who helped transition the course from Tensorflow to PyTorch.

## Chapter Outline

1. [Gradient Descent](chapters/chapter1_gradient-descent.ipynb)
2. [Stochastic Gradient Descent](chapters/chapter2_stochastic-gradient-descent.ipynb)
3. [Introduction to Pytorch & Neural Networks](chapters/chapter3_pytorch-neural-networks-pt1.ipynb)
4. [Training Neural Networks](chapters/chapter4_neural-networks-pt2.ipynb)
5. [Introduction to Convolutional Neural Networks](chapters/chapter5_cnns-pt1.ipynb)
6. [Advanced Convolutional Neural Networks](chapters/chapter6_cnns-pt2.ipynb)
7. [Advanced Deep Learning](chapters/chapter7_advanced-deep-learning.ipynb)

## Getting Started

The material on this site is written in Jupyter notebooks and rendered using [Jupyter Book](https://jupyterbook.org/intro.html) to make it easily accessible. However, if you wish to run these notebooks on your local machine, you can do the following:

1. Clone the GitHub repository:
   ```sh
   git clone https://github.com/TomasBeuzen/deep-learning-with-pytorch.git
   ```
2. Install the conda environment by typing the following in your terminal:
   ```sh
   conda env create -f dlwpt.yaml
   ```
3. Open the course in JupyterLab by typing the following in your terminal:
   ```sh
   cd deep-learning-with-pytorch
   jupyterlab
   ```

>If you're not comfortable with `git`, `GitHub` or `conda`, feel free to just read through the material on this website - you're not missing out on anything! 
