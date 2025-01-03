# Introduction

This project contains the python code used to get the results of my thesis - "Classification of Injury Risk in Football".

# Code Explanation
## "Functions" Folder
"Functions" folder contains some of the functions used in this work. Most of the functions are within their respective files (it usually goes in the following order: libraries, function definitions, rest of the code), however, specific functions that were used multiple times across multiple *.py* files are in the *Model_helpers.py*. These are created to save space in the modelling scripts, like reading the data, splitting the features and scaling the data, which is done for multiple data set variations that are compared in the thesis, thus saving a lot of space in the script. It also contains functions to plot the ROC curve and the confusion matrix, that, once again, are generally large and together take up 27 lines of code. They are used multiple times in each modelling script and helps save space when used this way.

## Prefix 1
Code used in scraping the data and creating the data set;

## Prefix 2
Code used in preprocessing the data set: creating derivative features, clean - up of corrupted data;

## Prefix 3
Code used in exploratory data analysis;

## Prefix 4
Code used to find and remove outliers;

## Prefix 5
Code used to undersample the data starting from different outlier removal approaches;

## Prefix 6
Code used in feature selection and multicollinearity elimination;

## Prefix 7
Code used to train and evaluate the models;

## Prefix 8
Code used to construct voting ensembles.
