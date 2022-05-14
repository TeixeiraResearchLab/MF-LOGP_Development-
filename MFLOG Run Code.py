"""
MFLOGP Prediction Script
NOTE: This script is the culmination of the MFLOGP project. Within this script
      a user can input a molecular formula and the algorithm will return its
      prediction. The user can either run a single compound within the script
      or load in a list of formulas in an excel (.xlsx) file. 
      
      It is important to note that the MFLOGP algorithm only works on ORGANIC
      compounds. Attempting to analyze an ingorangic compound will lead to error
      codes. 

Instructions:
1.) Add single compound or directory and file for list of formulas
2.) Choose either single compound or list options
3.) RUN!!

PI: Dr. Andrew Teixeira
Author: David Kenney ('25) 
Institution: Worcester Polytechnic Institute

For all questions please email arteixeira@wpi.edu
"""

#Load in Python packages
import numpy as np
import pandas as pd
import joblib
import chemparse as cp 
import sys

"""USER INPUTS"""
single_compound = 1
compound_list = 0

molecular_formula = 'C5H10' #"""Single Compound Only"""#

file_dir = r'"INPUT FILE DIRECTORY"' #"""Multi-compound"""#
sheetname = 'Sheet1'

#----------------------------------------------------------------------------#
if (single_compound) | ((single_compound !=1) & (compound_list !=1)) | (single_compound & compound_list):
    
    if ((single_compound !=1) & (compound_list !=1)):
        print('No run option chosen, default single compound was run')
        single_compound = 1
        
    if (single_compound & compound_list):
        print('Both run option chosen, default single compound was run')
        single_compound = 1
        compound_list = 0

    molecular_formula = pd.DataFrame([molecular_formula],columns = ['Formula'])
    
    features = molecular_formula['Formula'].apply(cp.parse_formula)
    features = pd.json_normalize(features)
    features = features.fillna(0)
    
    elements = ['C','H','N','O','S','P','F','Cl','Br','I']
    compound = pd.DataFrame(0,index = np.arange(1),columns = [elements])
    
    for ii in range(0,len(features.columns)):
        if features.columns[ii] == 'C':
            compound['C'] = features['C']
    
        elif features.columns[ii] == 'H':
            compound['H'] = features['H']
        
        elif features.columns[ii] == 'N':
            compound['N'] = features['N']
            
        elif features.columns[ii] == 'O':
            compound['O'] = features['O']
            
        elif features.columns[ii] == 'S':
            compound['S'] = features['S']
            
        elif features.columns[ii] == 'P':
            compound['P'] = features['P']
            
        elif features.columns[ii] == 'F':
            compound['F'] = features['F']
            
        elif features.columns[ii] == 'Cl':
            compound['Cl'] = features['Cl']
            
        elif features.columns[ii] == 'Br':
            compound['Br'] = features['Br']
            
        elif features.columns[ii] == 'I':
            compound['I'] = features['I']
            
        else:
            sys.exit('Incompatible Formula')   

elif compound_list:
    
    data = pd.read_excel(file_dir,sheet_name = sheetname)
    
    features = data['Formula'].apply(cp.parse_formula)
    features = pd.json_normalize(features)
    features = features.fillna(0)
    
    if len(features.columns) > 10:
        sys.exit('Incompatible Formula')   
    else:
        compound = features

          

MFLOGP = joblib.load(r'C:\Users\David\Worcester Polytechnic Institute (wpi.edu)\gr-TeixeiraLab - research-htl\Partition Coefficient Machine Learning\Final Project\MFLOGP.sav')
scale_X = joblib.load(r'C:\Users\David\Worcester Polytechnic Institute (wpi.edu)\gr-TeixeiraLab - research-htl\Partition Coefficient Machine Learning\Final Project\scale_X.sav')
scale_y = joblib.load(r'C:\Users\David\Worcester Polytechnic Institute (wpi.edu)\gr-TeixeiraLab - research-htl\Partition Coefficient Machine Learning\Final Project\scale_y.sav')
    
compound_prediction = scale_y.inverse_transform((MFLOGP.predict(pd.DataFrame(scale_X.transform(compound),columns = elements))).reshape(-1,1))


if single_compound:
    print(compound_prediction)
elif compound_list:
    predictions = compound_prediction
    print('Please see "predictions" in variable explorer for model predictions')






    
