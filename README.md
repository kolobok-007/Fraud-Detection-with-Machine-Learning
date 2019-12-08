# Description

This repository contains code for building a classifier model for determing persons of intrest (POIs) from [Enron scandal](https://en.wikipedia.org/wiki/Enron_scandal) legal case. The data used to train and test is based on email and financial data.

# Files
`tools/` contains function to prepare the data

`data/` contains supplementary data

`classifier_selection.py` contains code for feautre selection, feature scaling, classifier selection, paramter tuning and evalution of performance.

`tester.py` function for testing classifier

`.pkl` files are pickled files that contain classifier, dataset and feature_list
