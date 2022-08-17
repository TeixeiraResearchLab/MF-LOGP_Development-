"""
Elemental Partition Coefficient Machine Learning Program
NOTE: This script will analyze various regression techniques (i.e. linear
      random forest, etc.) Users can easily swap/add/remove regression
      functions as they see fit. To use this script, users must have their
      data in an excel (.xlsx) file format and the training compounds of
      interest must be broken apart into individual elemental features 
      (C,H,N,O,S,...)
      
      If you have a list of molecular formulas that need to be broken into the 
      proper format, please see the formula parsing script within the repo.

      If you decide to add or swap regression methods and would like to tune
      its hyperparameters, please see the grid search algorithm within the repo
      
      If you have additional features that you would like to try beyond what is
      listed you may add them in the same format under the logical statement 
      below 
      
Instructions:
1.) Add file directory for training data
2.) Choose training capabilites and error analysis
3.) Adjust models (optional))
3.) RUN!!

PI: Dr. Andrew Teixeira
Author: David Kenney ('25) 
Institution: Worcester Polytechnic Institute

For all questions please email arteixeira@wpi.edu
"""

#Load in Python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Models
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

#Import Extra Packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler

"""USER INPUTS"""
file_dir = r'Input File Path Here'
sheetname = 'Input Sheet Name Here'

"""Model Training Controls"""
addtnl_features = 0 # 1 - YES, 0 - NO
hyperparameters = 0 # 1 - YES, 0 - NO
cross_validation = 0 # 1 - YES, 0 - NO

"""Model Performance Parameters"""
rmse_ = 1 # 1 - YES, 0 - NO
mae_ = 1 # 1 - YES, 0 - NO
r2_score_ = 1 # 1 - YES, 0 - NO

#--------------------------------------------------------------------------#

x_columns = ['C','H','N','O','S','P','F','Cl','Br','I']
y_column = ['Exp logp']

print('Importing data...')
training_data = pd.read_excel(file_dir,sheet_name = sheetname) 

x = pd.DataFrame(training_data,columns = x_columns)
y = pd.DataFrame(training_data,columns = y_column)

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


if hyperparameters:
    print('Loading tuned models...')
    model1 = LinearRegression()
    model2 = Ridge(alpha = 10)
    model3 = Lasso(alpha = 0.0001)
    model4 = RandomForestRegressor(n_estimators = 1000, min_samples_split = 2, min_samples_leaf= 2)
    model5 = GradientBoostingRegressor(min_samples_split= 3,n_estimators = 500)
    model6 = KNeighborsRegressor(n_neighbors = 7, leaf_size = 4)

else:
    print('Loading base models...')
    model1 = LinearRegression()
    model2 = Ridge()
    model3 = Lasso()
    model4 = RandomForestRegressor()
    model5 = GradientBoostingRegressor()
    model6 = KNeighborsRegressor()



print('Separating training and testing data...')
[X, Vault_X, Y, Vault_Y] = train_test_split(x,y,train_size = 0.85, random_state = 42)
[X_train,X_test,y_train,y_test]=train_test_split(X[x_columns],Y,train_size=0.8,shuffle = True)


print('Normalizing data...')
scale_X = MinMaxScaler().fit(X_train)
scale_y = MinMaxScaler().fit(y_train)

nX_train = pd.DataFrame(scale_X.transform(X_train),columns = x_columns)
nX_test = pd.DataFrame(scale_X.transform(X_test),columns = x_columns)
ny_train = pd.DataFrame(scale_y.transform(y_train),columns = y_column)
ny_test = pd.DataFrame(scale_y.transform(y_test),columns = y_column)

if cross_validation:
    print('Training models...')
    kf = model_selection.KFold(n_splits = 10,shuffle = True)
    ii = 0
    for train_index, test_index in kf.split(nX_train):
        print(ii)
        X_train_cv = nX_train.iloc[train_index,:]
        y_train_cv = ny_train.iloc[train_index,:]
        model1.fit(X_train_cv,np.ravel(y_train_cv))
        model2.fit(X_train_cv,np.ravel(y_train_cv))
        model3.fit(X_train_cv,np.ravel(y_train_cv))
        model4.fit(X_train_cv,np.ravel(y_train_cv))
        model5.fit(X_train_cv,np.ravel(y_train_cv))
        model6.fit(X_train_cv,np.ravel(y_train_cv))
    
        ii += 1
else:
    print('Training models...')
    model1.fit(nX_train, np.ravel(ny_train))
    model2.fit(nX_train, np.ravel(ny_train))
    model3.fit(nX_train, np.ravel(ny_train))
    model4.fit(nX_train, np.ravel(ny_train))
    model5.fit(nX_train, np.ravel(ny_train))
    model6.fit(nX_train, np.ravel(ny_train))



if rmse_:
    print('Calculating root mean squared error...')
    model1_rmse = np.sqrt(mean_squared_error(y_test,scale_y.inverse_transform((model1.predict(nX_test)).reshape(-1,1))))
    model2_rmse = np.sqrt(mean_squared_error(y_test,scale_y.inverse_transform((model2.predict(nX_test)).reshape(-1,1))))
    model3_rmse = np.sqrt(mean_squared_error(y_test,scale_y.inverse_transform((model3.predict(nX_test)).reshape(-1,1))))
    model4_rmse = np.sqrt(mean_squared_error(y_test,scale_y.inverse_transform((model4.predict(nX_test)).reshape(-1,1))))
    model5_rmse = np.sqrt(mean_squared_error(y_test,scale_y.inverse_transform((model5.predict(nX_test)).reshape(-1,1))))
    model6_rmse = np.sqrt(mean_squared_error(y_test,scale_y.inverse_transform((model6.predict(nX_test)).reshape(-1,1))))
    rmse = np.array([model1_rmse,model2_rmse,model3_rmse,model4_rmse,model5_rmse,model6_rmse])
    
    print(rmse)

    plt.figure(1)
    plt.bar(['Linear','Ridge','Lasso','Random Forest','Gradient Boosted','KNN'],rmse)
    plt.xticks(rotation = 45,ha = 'right')

if mae_:
    print('Calculating mean absolute error...')
    model1_mae = mean_absolute_error(y_test,scale_y.inverse_transform((model1.predict(nX_test)).reshape(-1,1)))
    model2_mae = mean_absolute_error(y_test,scale_y.inverse_transform((model2.predict(nX_test)).reshape(-1,1)))
    model3_mae = mean_absolute_error(y_test,scale_y.inverse_transform((model3.predict(nX_test)).reshape(-1,1)))
    model4_mae = mean_absolute_error(y_test,scale_y.inverse_transform((model4.predict(nX_test)).reshape(-1,1)))
    model5_mae = mean_absolute_error(y_test,scale_y.inverse_transform((model5.predict(nX_test)).reshape(-1,1)))
    model6_mae = mean_absolute_error(y_test,scale_y.inverse_transform((model6.predict(nX_test)).reshape(-1,1)))
    mae = np.array([model1_mae,model2_mae,model3_mae,model4_mae,model5_mae,model6_mae])
    
    print(mae)

    plt.figure(2)
    plt.bar(['Linear','Ridge','Lasso','Random Forest','Gradient Boosted','KNN'],mae)
    plt.xticks(rotation = 45,ha = 'right')

if r2_score_:
    print('Calculating the r2 score...')
    model1_r2 = r2_score(y_test,scale_y.inverse_transform((model1.predict(nX_test)).reshape(-1,1)))
    model2_r2 = r2_score(y_test,scale_y.inverse_transform((model2.predict(nX_test)).reshape(-1,1)))
    model3_r2 = r2_score(y_test,scale_y.inverse_transform((model3.predict(nX_test)).reshape(-1,1)))
    model4_r2 = r2_score(y_test,scale_y.inverse_transform((model4.predict(nX_test)).reshape(-1,1)))
    model5_r2 = r2_score(y_test,scale_y.inverse_transform((model5.predict(nX_test)).reshape(-1,1)))
    model6_r2 = r2_score(y_test,scale_y.inverse_transform((model6.predict(nX_test)).reshape(-1,1)))
    r2 = np.array([model1_r2,model2_r2,model3_r2,model4_r2,model5_r2,model6_r2])
    
    print(r2)    

    plt.figure(3)
    plt.bar(['Linear','Ridge','Lasso','Random Forest','Gradient Boosted','KNN'],r2)
    plt.xticks(rotation = 45,ha = 'right')


print('Plotting predictions...')
plt.figure(4)
plt.scatter(y,scale_y.inverse_transform((model1.predict(pd.DataFrame(scale_X.transform(x[x_columns]),columns = x_columns))).reshape(-1,1)))
plt.xlabel('Known LOGP')
plt.ylabel('Predicted LOGP')
plt.gca().autoscale_view()

plt.figure(5)
plt.scatter(y,scale_y.inverse_transform((model2.predict(pd.DataFrame(scale_X.transform(x[x_columns]),columns = x_columns))).reshape(-1,1)))
plt.xlabel('Known LOGP')
plt.ylabel('Predicted LOGP')
plt.gca().autoscale_view()

plt.figure(6)
plt.scatter(y,scale_y.inverse_transform((model3.predict(pd.DataFrame(scale_X.transform(x[x_columns]),columns = x_columns))).reshape(-1,1)))
plt.xlabel('Known LOGP')
plt.ylabel('Predicted LOGP')
plt.gca().autoscale_view()

plt.figure(7)
plt.scatter(y,scale_y.inverse_transform((model4.predict(pd.DataFrame(scale_X.transform(x[x_columns]),columns = x_columns))).reshape(-1,1)))
plt.xlabel('Known LOGP')
plt.ylabel('Predicted LOGP')
plt.gca().autoscale_view()

plt.figure(8)
plt.scatter(y,scale_y.inverse_transform((model5.predict(pd.DataFrame(scale_X.transform(x[x_columns]),columns = x_columns))).reshape(-1,1)))
plt.xlabel('Known LOGP')
plt.ylabel('Predicted LOGP')
plt.gca().autoscale_view()

plt.figure(9)
plt.scatter(y,scale_y.inverse_transform((model6.predict(pd.DataFrame(scale_X.transform(x[x_columns]),columns = x_columns))).reshape(-1,1)))
plt.xlabel('Known LOGP')
plt.ylabel('Predicted LOGP')
plt.gca().autoscale_view()
