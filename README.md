# MF-LOGP_Development

This repository will hold all relavent scripts and files for the MF-LOGP Algorithm and will work as an extension to the supplementary information for "Developing a Dimensionally Reduced Machine Learning Model for Predicting Single Component Octanol-Water Partition Coefficients"

Within the repository you'll find:
  File:
  - MFLOGP Data Set.xlxs: Entire data set used to train/test/validate the MF-LOGP algorithm
  - MFLOGP Run Code.py: Top level script to use the data set to train and test various model performances
  - MFLOGP Formula Parser: Pyhton script that will allow users to parse new formulas with ease
  - MFLOGP Hyperparameter Tuning Script: Users can use this script to tune hyperparameters for new models
  - MFLOGP Run Code: Here users can test a single formula or list of compounds and eaily predict partition coefficients

  Variables:
  - MFLOGP.sav: Exported final version of the MF-LOGP algorithm
  - scale_X.sav: Information required to normalize new compounds
  - scale_y.sav: Information required to normalize new predictions

Important Notes:
1. Due to size constraints, the MFLOGP.sav is located in a compressed zip folder
2. Before using any of these files, users will need to edit the file directories at the beginnging of the scripts. 

For all questions please contact:
Dr. Andrew Teixeira (arteixeira@wpi.edu) and David Kenney (dhkenney@wpi.edu)
