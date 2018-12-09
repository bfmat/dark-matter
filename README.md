# Machine Learning for Dark Matter Detection
I present novel deep learning algorithms that perform significantly better than previous research at separating WIMP dark matter candidate events from background radiation. In the PICO-60 and DEAP-3600 dark matter experiments, alpha particles make up a major and difficult-to-separate class of background radiation. Developing a discriminator function manually is time-consuming and slows down research.

Machine learning is a powerful solution to this problem. However, impure calibration data (in the PICO-60 experiment) and a spherical detector format (in DEAP-3600) pose challenges to applications of neural networks. In this study, I have overcome these issues by developing semi-supervised learning algorithms that alleviate data impurity, improving accuracy from 80.7% to 99.2% in PICO-60. I have also developed new convolutional neural network processes that more effectively handle the spherical detector, reducing the number of false positives from 91% to 75.7% in DEAP-3600. 

## Technologies
This repository contains Python 3 code. NumPy, SciPy, scikit-image, scikit-learn, Matplotlib, Keras, and PyROOT are used. To execute scripts, first add the root folder of this repository to your `$PYTHONPATH`.
