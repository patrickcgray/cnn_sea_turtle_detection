# cnn_sea_turtle_detection

[![DOI](https://zenodo.org/badge/158115622.svg)](https://zenodo.org/badge/latestdoi/158115622)

## Code for Methods in Ecology and Evolution paper: "A Convolutional Neural Network for Detecting Sea Turtles in Drone Imagery"

#### Paper can be accessed at: https://doi.org/10.1111/2041-210X.13132

### Using this code:

Running run.sh in bash will run the full turtle detection workflow.

#### Notes:
* Python 2.7 is required and nonstandard python packages necessary are: numpy, scipy, keras, tables, and hdf5storage
* This setup runs on preprocessed imagery contained in the .mat file. Full turtle image data along with labels for independent machine learning development can be found at doi:10.5061/dryad.5h06vv2

#### File Details:
* data.py                 
  * defines utility functions for model creation and matlab ingestion functions
* cnn_predict_stack.py
  * run prediction on processed images
* DukeTurtle_info.h5
  * Trained model weights file
* DukeTurtle_info.json
  * Model definition file
* DukeTurtle_test.mat
  * processed and tiled RGB image data that is fed into the model. Training / validation split is 85% train / 15% validation 
