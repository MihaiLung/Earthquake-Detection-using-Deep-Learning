# Earthquake-Detection-using-Deep-Learning

## Introduction
This code was designed for the task of predicting the timing of an earthquake from a short sequence of a very high-frequency accoustic signal time series.

## Approach
My approach combines deep learning with manual feature engineering. One of the inputs to the model is the raw data itself, but also manually designed rolling local features of the data, which are also then fed through a deep learning model, making this a hybrid approach.

## Basic algorithm functionality
For each sub-model, the algorithm functions as follows:
 1. Manual input generation - unless the raw data is the input for this sub-model, this is either a transformation of the raw input data or rolling statistics computed locally on the input.
 2. Convolutional Neural Network - the input data is first fed to a convolutional neural network, which extracts local features across the input.
 3. De-sequencing the information. Two main approaches were adopted for translating the features from a matrix of local feature vectors to a single globally-representative feature vector
    a. Long Short-Term Memory Recurrent Neural Network - the features extracted by the CNN fed to a bi-directional recurrent network, which extracts sequential information contained in the features.
    b. Hierarchical features - the features extracted by the CNN are gradually pooled into one single vector describing the entire sequence.
 
## Models
Based on the building-block sub-models as described above, I developed a number of models trying different approaches at solving this challenge. Included is a model file that exemplifies the functioning of the algorithm. More optimized models will be added once the competition ends.

## Files
1. **LayerGenerator**

Flexible end-to-end module generation utility. The program defines generator objects which can be used efficiently to create similar complex modules adapted for the characteristics of the input data. The aim is to minimise the code required in the model program and to allow for quick tweaks and clear visualisation of the model's architecture.
