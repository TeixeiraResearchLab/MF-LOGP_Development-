"""
Machine learning hyperparameter tuning script

NOTE: This script will use training data to tune model specific hyperparameters
      for the models built into the MFLOGP Training Script.py file. 
      Users can easily swap/add/remove regression functions as they see fit. 
      To use this script, users must have their data in an excel (.xlsx) file 
      format and the training compounds of interest must be broken apart into 
      individual elemental features (C,H,N,O,S,...)
      
      If you have a list of molecular formulas that need to be broken into the 
      proper format, please see the formula parsing script within the repo.
      
      If you have additional features that you would like to try beyond what is
      listed you may add them in the same format under the logical statement 
      below 
      
Instructions:
1.) Add file directory for training data
2.) Choose file capabilites
3.) Adjust models (optional)
3.) RUN!!

PI: Dr. Andrew Teixeira
Author: David Kenney ('25) 
Institution: Worcester Polytechnic Institute

For all questions please email arteixeira@wpi.edu
"""
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split, GridSearchCV

#----------------------------------------------------------------------------#
"""User Inputs"""
file_dir = r'Input File Path Here'
sheetname = 'Input Sheet Name Here'

addtnl_features = 0 # 1 - YES, 0 - NO

ridge = 1 # 1 - YES, 0 - NO
lasso = 1 # 1 - YES, 0 - NO
random_forest = 1 # 1 - YES, 0 - NO
gradiend_boosted = 1 # 1 - YES, 0 - NO
neighbors = 1 # 1 - YES, 0 - NO

#----------------------------------------------------------------------------#

data = pd.read_excel(file_dir,sheet_name = sheetname) 

x_columns = ['C','H','N','O','S','P','F','Cl','Br','I']
y_column = ['Exp logp']

x = pd.DataFrame(data,columns = x_columns)
y = pd.DataFrame(data,columns = y_column)

if addtnl_features:
    print('Developing New Features...')
    MW =([(x['C']*12.0096)+(x['H']*1.00784)+(x['N']*14.00642)+(x['O']*15.99903)+(x['S']*32.059)+(x['P']*30.97376)+(x['F']*18.9984)+(x['Cl']*35.446)+(x['Br']*35.446)+(x['I']*126.9045)])
    DBE = ([x['C'] - (x['H']/2) + (x['N']/2) + 1])
    HC = ([x['H']/x['C']])
    NC = ([x['N']/x['C']])
    OC = ([x['O']/x['C']])
    SC = ([x['S']/x['C']])
    PC = ([x['P']/x['C']])
    FC = ([x['F']/x['C']])
    ClC = ([x['Cl']/x['C']])
    BrC = ([x['Br']/x['C']])
    IC = ([x['I']/x['C']])
    
    x['MW'] = pd.DataFrame([MW[0]]).transpose()
    x['DBE'] = pd.DataFrame([DBE[0]]).transpose()
    x['H/C'] = pd.DataFrame([HC[0]]).transpose()
    x['N/C'] = pd.DataFrame([NC[0]]).transpose()
    x['O/C'] = pd.DataFrame([OC[0]]).transpose()
    x['S/C'] = pd.DataFrame([SC[0]]).transpose()
    x['P/C'] = pd.DataFrame([PC[0]]).transpose()
    x['F/C'] = pd.DataFrame([FC[0]]).transpose()
    x['Cl/C'] = pd.DataFrame([ClC[0]]).transpose()
    x['Br/C'] = pd.DataFrame([BrC[0]]).transpose()
    x['I/C'] = pd.DataFrame([IC[0]]).transpose()

    x_columns = ['C','H','N','O','S','P','F','Cl','Br','I','MW','DBE','H/C','N/C','O/C','S/C','P/C','F/C','Cl/C','Br/C','I/C']

print('Separating training and testing data...')
[X, Vault_X, Y, Vault_Y] = train_test_split(x,y,train_size = 0.85, random_state = 42, shuffle = True)
[X_train,X_test,y_train,y_test]=train_test_split(X[x_columns],Y,train_size=0.8,shuffle = True)

if ridge:
    model = Ridge()
    parameters = {'alpha':[0.0001,0.001,0.01,0.1,1,2,5,10,20,30,40]}

    clf = GridSearchCV(model, parameters, verbose=10,cv = 8)
    clf.fit(X, np.ravel(Y))

    ridge_best_param = clf.best_params_
    ridge_best_score = clf.best_score_
    
if lasso:
    model = Lasso()
    parameters = {'alpha':[0.000001,0.00001,0.0001,0.001,0.01,0.1,1,2,5,10,20,30,40]}
    
    clf = GridSearchCV(model, parameters, verbose=10,cv = 8)
    clf.fit(X, np.ravel(Y))

    lasso_best_param = clf.best_params_
    lasso_best_score = clf.best_score_

if random_forest:
    model = RandomForestRegressor()
    parameters = {'n_estimators':[400, 600, 800, 1000,1200],'min_samples_split': [2, 4,6,8,10],'min_samples_leaf':[1,2,3,4,5]}
    
    clf = GridSearchCV(model, parameters, verbose=10,cv = 8)
    clf.fit(X, np.ravel(Y))

    rf_best_param = clf.best_params_
    rf_best_score = clf.best_score_
    
if gradiend_boosted:
    model = GradientBoostingRegressor()
    parameters = {'min_samples_split':[1,2,3,4,5,6,7,8,9,10,15,20,25,30],'n_estimators': [100,200,300,400,500]}

    clf = GridSearchCV(model, parameters, verbose=10,cv = 8)
    clf.fit(X, np.ravel(Y))

    gb_best_param = clf.best_params_
    gb_best_score = clf.best_score_
    
    
if neighbors:
    model = KNeighborsRegressor()
    parameters = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10,15,20,25,30],'leaf_size': [1,2,3,4,5,6,7,8,9,10,15,20,25,30]}
    
    clf = GridSearchCV(model, parameters, verbose=10,cv = 8)
    clf.fit(X, np.ravel(Y))

    knn_best_param = clf.best_params_
    knn_best_score = clf.best_score_
    
    



