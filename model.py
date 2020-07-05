
import pandas as pd
import pickle
import joblib

df=pd.read_csv('dataset.csv')
df.head()
# Note : Team 1 is the batting team and team 2 is the bowling team


# Generating some more feature from my domain knowledge that can help the model.
df['Run_rate']=df['Run_Scored']/df['over_bowled']
df['Wickets_Remaining']=10-df['Wickets_fallen']
df['Accelerating_Factor'] = 4*(df['over_bowled'])+(df['Wickets_Remaining'])**3
df['Bat_Depth'] = 2
df.loc[df['Wickets_Remaining']>=7,'Bat_Depth'] = 10
df.loc[(df['Wickets_Remaining']==5)|(df['Wickets_Remaining']==6),'Bat_Depth'] = 8
df.loc[(df['Wickets_Remaining']==4)|(df['Wickets_Remaining']==3),'Bat_Depth'] = 5




# Encoding the teams according to their past 15 years performances
team_rank={'team1_name': { 'Zimbabwe': 12, 'Scotland':17, 'Afghanistan':11, 'Australia':1, 'Pakistan':7,
       'New Zealand':6, 'Bangladesh':10, 'India':3, 'South Africa':2, 'England':4,
       'Papua New Guinea':23, 'Hong Kong':20, 'Ireland':13, 'West Indies':9,
       'Sri Lanka':8, 'United Arab Emirates':18, 'Nepal':16, 'Namibia':24,
       'United States of America':25, 'Oman':21, 'Bermuda':19, 'Kenya':15, 'Canada':22,
       'Netherlands':14, 'Asia XI':5},
        'team2_name': { 'Zimbabwe': 12, 'Scotland':17, 'Afghanistan':11, 'Australia':1, 'Pakistan':7,
       'New Zealand':6, 'Bangladesh':10, 'India':3, 'South Africa':2, 'England':4,
       'Papua New Guinea':23, 'Hong Kong':20, 'Ireland':13, 'West Indies':9,
       'Sri Lanka':8, 'United Arab Emirates':18, 'Nepal':16, 'Namibia':24,
       'United States of America':25, 'Oman':21, 'Bermuda':19, 'Kenya':15, 'Canada':22,
       'Netherlands':14, 'Africa XI':5} }
df.replace(team_rank,inplace=True)



# splitting data into target and features
target=df['Inning_Runs']
features=df.drop(['Inning_Runs'],axis=1)
features=features.drop(['Wickets_fallen'],axis=1)



# Hyper-Parameter tunning using GridSearchCV
"""
from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 
param_grid = {
  
    'max_depth': [10,12,14],
    'min_samples_leaf': [6,8],
    'min_samples_split': [2],
    'n_estimators': [180,200,220]
}
# Create a based model
rf1 = RandomForestRegressor( random_state=42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf1, param_grid = param_grid, 
                          cv =5, verbose = 2,scoring='neg_mean_absolute_error')
# Fit the grid search to the data
grid_search.fit(features, target)
grid_search.best_params_
"""


# splitting data into test and train for validation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestRegressor
Reg = RandomForestRegressor(max_depth=12,min_samples_leaf=8,min_samples_split=2,n_estimators=250, random_state=42)
Reg.fit(X_train, y_train)





# Checking the model
from sklearn.metrics import mean_absolute_error
y_pred=Reg.predict(X_test)
print(y_pred)
print(y_test.head())

mae=mean_absolute_error(y_test,y_pred)
print("Model mae is :",mae)


# Creating final model with best hyper-parameters and training on full data
Final_Reg = RandomForestRegressor(max_depth=12,min_samples_leaf=8,min_samples_split=2,n_estimators=250, random_state=42)
Final_Reg.fit(features,target)



#saving model to pickle
filename="finalized_model.sav"
joblib.dump(Final_Reg,filename,compress=3)


