# Masters-Dissertation

Making a GAN to map minerals using a pipeline produced by Saranath and parente 2021.
Utilising labelled data by Plebani et al 2022

## Making a Map
The main entry to the mapping procedure is in the Auto_mapping_ratio/Map_creating.ipynb
Here the image you want to map can be added as a .lbl file into the Basemap directory

## ROC method
The ROC method is found in the ROC_validation directory, it requires the labelled dataset from Plebani et al 2022 to be used: https://github.com/Banus/crism_ml

## Pre_Processing
The creates the training data from the CRISM images. Add CAT cleaned CRISM.hdr files to the directory

## PYtorch_model
Contains the WGAN pytorch model

## Exemplar_library
Contains the code to turn the labelled dataset to a representation  it requires the labelled dataset from Plebani et al 2022 to be used: https://github.com/Banus/crism_ml
