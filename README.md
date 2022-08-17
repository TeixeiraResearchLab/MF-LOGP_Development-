# MF-LOGP_Development

This repository holds all relavent scripts and files for the MF-LOGP Algorithm and acts as an extension to the supplementary information for "Dimensionally Reduced Machine Learning Model for Predicting Single Component Octanol-Water Partition Coefficients". 

The code for this project was written in Python through the Anaconda environment. It is recomended that the code be used in this same environment but is versatile enough to be used in terminal-based Python. Instructions for using using the MF-LOGP algorithm in Anaconda and a Windows Command Prompt will be included below. 

Within the repository you'll find:
  Instructions:
  - Running MFLOGP via Command Prompt: Instructions that detail how to download Python, MFLOGP code and data, and run via command prompt
  - Running MFLOGP via Anaconda Terminal: Instructions that detail how to download Anaconda, MFLOGP code and data, and run via Spyder

  File:
  - MFLOGP Data Set.xlxs: Entire data set used to train/test/validate the MF-LOGP algorithm
  - MFLOGP Multicompound List.xlsx: Example list of compounds that can be used with the multi-compound prediction capabilities of MF-LOGP
  - MFLOGP_Run_Code.py: Top level script to use the data set to train and test various model performances
  - MFLOGP_Formula_Parser: Pyhton script that will allow users to parse new formulas with ease
  - MFLOGP_Hyperparameter Tuning Script: Users can use this script to tune hyperparameters for new models
  - MFLOGP_Training_Script: Users can take their data and train de-featured models

  Variables:
  - MFLOGP.sav: Exported final version of the MF-LOGP algorithm
  - scale_X.sav: Information required to normalize new compounds
  - scale_y.sav: Information required to normalize new predictions

For all questions please contact:
Dr. Andrew Teixeira (arteixeira@wpi.edu) and David Kenney (dhkenney@wpi.edu)
