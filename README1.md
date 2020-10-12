# py-xvis

Python implementation for XVis Toolbox release with the book Computer 
Vision for X-Ray Testing. Originally implemented in Matlab by Domingo Mery for 
first edition of the book.

# Requirements

- Python 3.6 or higher
- Numpy
- Scipy
- Matplotlib
- OpenCV 4.0 or higher

# Instalation
In the first installation use directly in the directory folder:

`pip install .`

Then, to upgrade:

`pip install --upgrade .`





# Google Colab

All examples of the Book have been implemented in Jupyter Notebooks tha run on Google Colab.


## Chapter 01: X-ray Testing 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1meevAFZkZ_0SUBPYqo_nl7grBcqfeVMP?usp=sharing)
* Example 1.1: Displaying X-ray images
* Example 1.2: Dual Energy
* Example 1.3: Help of PyXvis functions


## Chapter 02: Images for X-ray Testing 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MDc5wbH11I5PV18l5ZSt92yugz0eHi4W?usp=sharing)
* Example 2.1: Displaying an X-ray image of GDXray


## Chapter 03: Geometry in X-ray Testing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/14CgSCGcpm4Wc4SbVB70iBx6Cw6rXoZAq?usp=sharing)

* Example 3.1: Euclidean 2D transformation
* Example 3.2: Euclidean 3D transformation
* Example 3.3: Perspective projection
* Example 3.4: Cubic model for distortion correction
* Example 3.5: Hyperbolic model for imaging projection
* Example 3.6: Geometric calibration
* Example 3.7: Epipolar geometry
* Example 3.8: Trifocal geometry
* Example 3.9: 3D reconstruction


## Chapter 04: X-ray Image Processing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1edbph3qfb2PfKOWPGUWUsC0DeX8muO9U?usp=sharing)

* Example 4.1: Aritmetic average of images
* Example 4.2: Contrast enhancement
* Example 4.3: Shading correction
* Example 4.4: Detection of defects using median filtering
* Example 4.5: Edge detection using gradient operation
* Example 4.6: Edge detection with LoG
* Example 4.7: Segmentation of bimodal images
* Example 4.8: Welding inspection using adaptive thresholding
* Example 4.9: Region growing
* Example 4.10: Defects detection using LoG approach
* Example 4.11: Segmentation using MSER
* Example 4.12: Image restoration


## Chapter 05: X-ray Image Representation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dwGTGHA1CR1om3MirGX5VCVhQgVc-g3-?usp=sharing)

* Example 5.1: Geometric features
* Example 5.2: Elliptical features
* Example 5.3: Invariant moments
* Example 5.4: Intenisty features
* Example 5.5: Defect detection usin contrast features
* Example 5.6: Crossing line profiles (CLP)
* Example 5.7: SIFT
* Example 5.8: feature se;ection
* Example 5.9: Example using intenisty features
* Example 5.10: Example using geometric features


## Chapter 06: Classification in X-ray Testing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zGx0HpAt7EtOiORXkTluOPDW4w5alNSj?usp=sharing)

* Example 6.1: Basic classification example
* Example 6.2: Minimal distance (dmin)
* Example 6.3: Bayes
* Example 6.4: Mahalanobis, LDA and QDA
* Example 6.5: KNN
* Example 6.6: Neural networks
* Example 6.7: Support Vector Machines (SVM)
* Example 6.8: Training and testing many classifiers
* Example 6.9: Hold-out
* Example 6.10: Cross-validation
* Example 6.11: Confusion matrix
* Example 6.12: ROC and Precision-Recall curves
* Example 6.13: Example with intensity features
* Example 6.14: Example with geometric features


## Chapter 07: Deep Learing in X-ray Testing

* Example 7.1: Basic neural networks (from skratch)
* Example 7.2: Neural network using sklearn [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ohs0hBDu5zRtNagbqBCJV6fmxq63CxS6?usp=sharing)

* Example 7.3: Convolutional Neural Network [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nI3AABdBJKdT680L-ouUwX3ywpajv8bC?usp=sharing)

* Example 7.4: Pre-trained models [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JA3sgXqDHN7gkAdv1dRa-a-IgsArAA2M?usp=sharing)

* Example 7.5: Fine tunning [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iC_XLsyBru3I2RpJot8YCGt_AbQNw3mz?usp=sharing)

* Example 7.6: Generative Adversarial Networks (GANs) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Bv9wptpLuxjXxcx6UQmPGtLdZvx949iU?usp=sharing)


## Chapter 08: Simulation in X-ray Testing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1s7bKncSrQhIq_kW0qO3JvUOyyK8rfp3Q?usp=sharing)

* Example 8.1: Basic simulation using voxels
* Example 8.2: Simulation of defects using mask
* Example 8.3: Simulation of ellipsoidal defects
* Example 8.4: Superimposition of threat objects


## Chapter 09: Applications in X-ray Testing

* Example 9.1: Defect detection in castings [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FLyUEYrevSu3RbZQaoPsd2BMG4MvRew0?usp=sharing)

* Example 9.2: Defect detection in welds [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mFiaoEsuhAEQoev_jgPEv35G1lIt55F8?usp=sharing)







[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]


























# GDXray database
In our examples, we use [GDXray](https://domingomery.ing.puc.cl/material/gdxray/). The X-ray images included in GDXray+ can be used free of charge, for research and educational purposes only. Redistribution and commercial use is prohibited. 

Mery, D.; Riffo, V.; Zscherpel, U.; Mondragón, G.; Lillo, I.; Zuccar, I.; Lobel, H.; Carrasco, M. (2015): [GDXray: The database of X-ray images for nondestructive testing](http://dmery.sitios.ing.uc.cl/Prints/ISI-Journals/2015-JNDE-GDXray.pdf). Journal of Nondestructive Evaluation, 34.4:1-12.
