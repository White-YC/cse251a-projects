# CSE 251A Projects (UC San Diego, Winter 2026)

My three programming projects for CSE 251A, ML: Learning Algorithms. Each folder holds my own code and, where available, my written report. Course handouts, starter data and pretrained weights are not included.

## Project 1: Prototype selection for 1-NN with Learning Vector Quantization

`project1-prototype-selection/`

Selects a small set of prototypes for a 1-nearest-neighbor classifier on MNIST using LVQ and LVQ3, in both prototype-generation and prototype-selection form, and compares them with random selection. With 500 prototypes, LVQ3 reaches 0.946 test accuracy against 0.851 for random selection.

## Project 2: Coordinate descent for logistic regression

`project2-coordinate-descent/`

Implements Gauss-Southwell coordinate descent with a per-coordinate Newton step for logistic regression on the UCI Wine data (first two classes), compares it with random coordinate selection, and adds a sparse variant with hard thresholding.

## Project 3: Adversarial examples for neural networks

`project3-adversarial-examples/`

Implements FGSM and momentum iterative FGSM (MI-FGSM) against a pretrained three-hidden-layer MNIST classifier, with gradients back-propagated by hand in NumPy. At epsilon 0.10, FGSM lowers accuracy to 0.146 and MI-FGSM to 0.115. The script expects the course-provided test data and network weights in a `data/` folder, which are not part of this repository.
