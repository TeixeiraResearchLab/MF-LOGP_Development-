# -*- coding: utf-8 -*-
"""
Molucular Formula Parser

NOTE: This script take a given group of molecular formulas and create a
      dataframe of elemental compositions for each elements. If this is the 
      first time using this script of the chemparse library run the following
      command in your command window:
    
      --> pip install chemparse
      
      For more information please read the latest documentation:
      https://pypi.org/project/chemparse/
      
      To use this script, users must have their data in an excel (.xlsx) file 
      format with all formulas in a single column with the header "Formula"
            
      The scope of this project was to analyze organic compounds, thus
      inorganic compounds were removed. Users can add and remove constraints
      as they see fit
      
Instructions:
1.) Add file directory for training data
2.) Adjust data constraints
3.) RUN!!

PI: Dr. Andrew Teixeira
Author: David Kenney ('25) 
Institution: Worcester Polytechnic Institute

For all questions please email arteixeira@wpi.edu
"""
import pandas as pd
import chemparse as cp

#----------------------------------------------------------------------------#

"""User Inputs"""

file_dir = r'C:\Users\David\Worcester Polytechnic Institute (wpi.edu)\gr-TeixeiraLab - research-htl\Partition Coefficient Machine Learning\Paper Training Sets\new setv2.xlsx'
sheetname = 'Functional Groups'

#----------------------------------------------------------------------------#

data = pd.read_excel(file_dir,sheet_name = sheetname) 

features = data['Formula'].apply(cp.parse_formula)
features = pd.json_normalize(features)
features = features.fillna(0)

full_data = data.join(features)

full_data = full_data[(full_data['Exp logp']<30) &(full_data['C']>0) & (full_data['Si']==0)  & (full_data['Si']==0) & (full_data['As']==0)  & (full_data['Hg']==0) & (full_data['Sn']==0)  
                    & (full_data['B']==0) & (full_data['Se']==0) & (full_data['Na']==0) ]

full_data = full_data[['Names','Formula','CAS No','SMILES','Exp logp', 'Reference','C','H','N','O','S','P','F','Cl','Br','I']]















