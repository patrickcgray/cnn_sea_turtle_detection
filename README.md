# cnn_sea_turtle_detection

[![DOI](https://zenodo.org/badge/158115622.svg)](https://zenodo.org/badge/latestdoi/158115622)

### Code for Methods in Ecology and Evolution paper: "A Convolutional Neural Network for Detecting Sea Turtles in Drone Imagery"

Turtle image data along with labels for independent machine learning development can be found at doi:10.5061/dryad.5h06vv2

#### Using this code:

Running run.sh in bash will run the full script turtle detection workflow.

* data.py                 
  * defines utility functions for model creation and matlab ingestion functions
* cnn_predict_stack.py    
  * run the prediction on processed images
* DukeTurtle_info.h5
  * Trained model weights file
* DukeTurtle_info.json
  * Model definition file
* DukeTurtle_info.mat
  * processed images for training
* DukeTurtle_test.mat
  * processed images for testing
