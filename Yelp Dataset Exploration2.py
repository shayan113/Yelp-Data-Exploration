
# coding: utf-8

# In[56]:


# package imports
#basics
import numpy as np
import pandas as pd
import ast

#misc
import gc
import time
import warnings


#viz
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.gridspec as gridspec 
import matplotlib.gridspec as gridspec 

#settings
start_time=time.time()
color = sns.color_palette()
sns.set_style("dark")
import re

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.grid_search import GridSearchCV

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import mean_squared_error
from math import sqrt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importing business dataset
business=pd.read_csv("yelp_academic_dataset_business.csv")
end_time=time.time()
print("Took",end_time-start_time,"s")


# In[3]:


#take a peak
business.head()


# In[4]:


business.info()


# In[5]:


#Get the distribution of the ratings
x=business['stars'].value_counts()
x=x.sort_index()
#plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Star Rating Distribution")
plt.ylabel('# of businesses', fontsize=12)
plt.xlabel('Star Ratings ', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


# In[6]:


#Get the distribution of the citys
x=business['city'].value_counts()
x=x.sort_values(ascending=False)
x=x.iloc[0:20]
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8,color=color[3])
plt.title("Which city has the most reviews?")
locs, labels = plt.xticks()
plt.setp(labels, rotation=45)
plt.ylabel('# businesses', fontsize=12)
plt.xlabel('City', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


# In[7]:


type(business['categories'])


# In[8]:


print(business['categories'].head())


# In[9]:


# What are the popular business categories?
strinfo = re.compile(' ')
business_cat= business['categories'].str.cat(sep = ',')
business_cats = strinfo.sub('', business_cat)
#print(business_cats)
cats=pd.DataFrame(business_cats.split(','),columns=['category'])
print(cats.head())
x=cats.category.value_counts()
print(x.head(20))
print("There are ",len(x)," different types/categories of Businesses in Yelp!")
#prep for chart
x=x.sort_values(ascending=False)
x=x.iloc[0:20]


# In[10]:


#chart
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8)#,color=color[5])
plt.title("What are the top categories?",fontsize=25)
locs, labels = plt.xticks()
plt.setp(labels, rotation=80)
plt.ylabel('# businesses', fontsize=12)
plt.xlabel('Category', fontsize=12)
#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()


# In[11]:


business_NLasVeg = business[business['city']=='North Las Vegas']
print(business_NLasVeg.head())
print(business_NLasVeg.info())


# In[12]:


x=business['city'].value_counts()
print(x.head(20))


# In[13]:


#change all the nan value into 0
business_NLasVeg = business_NLasVeg.fillna(0).apply(pd.to_numeric, errors = 'ignore')
print(business_NLasVeg.head())
print(business_NLasVeg['attributes.Ambience'].head())


# In[14]:


# check that all nulls are removed
business_NLasVeg.isnull().sum().sum()


# In[15]:


sns.boxplot(x='attributes.WiFi', y='stars', data=business_NLasVeg);


# In[16]:


print(business_NLasVeg['categories'].head())


# In[17]:


business_NLasVeg.columns


# In[18]:


business_NLasVeg["attributes.RestaurantsAttire"].head(10)
#type(business_NLasVeg["review_count"])


# In[19]:


#split categories columns into sub category column
business_NLasVeg['categories_clean'] = list(map(lambda x: ''.join(x.split()),business_NLasVeg['categories'].astype(str)))
#print(business_NLasVeg['categories_clean'])
categories_df = business_NLasVeg.categories_clean.str.get_dummies(sep=',')

business_NLasVeg = business_NLasVeg.merge(categories_df, left_index=True, right_index=True)
print(business_NLasVeg.info())
#print(business_NLasVeg.head())
print(categories_df.columns)


# In[20]:


select_df = business_NLasVeg[business_NLasVeg['Restaurants'] == 1]
print(categories_df['Restaurants'].value_counts())
#print(select_df.head())

select_df = select_df.drop(columns = ['attributes'])
select_df.iloc[:, 40:60].head(20)
#print(select_df.info())


# In[21]:


print(select_df.info())


# In[22]:


print(select_df.columns)


# In[23]:


# columns with non-boolean categorical values:
cols_to_split = ['attributes.AgesAllowed', 'attributes.Alcohol', 'attributes.BYOBCorkage', 
                 'attributes.NoiseLevel', 'attributes.RestaurantsAttire', 'attributes.Smoking', 'attributes.WiFi']
new_cat = pd.concat([pd.get_dummies(select_df[col], prefix=col, prefix_sep='_') for col in cols_to_split], axis=1)
# keep all columns (not n-1) because 0's for all of them indicates that the data was missing (useful info)
select_df = pd.concat([select_df, new_cat], axis=1)
select_df.drop(cols_to_split, inplace=True, axis=1)
select_df.head()


# In[24]:


print(select_df.info())


# In[25]:


#drop columns with non-numeric
select_df = select_df.select_dtypes(exclude=['object'])
print(select_df.columns)


# In[26]:


print(select_df.info())


# In[27]:


y = select_df['stars']
X = select_df.drop(columns = ['stars'])


# In[28]:


sns.distplot(y, kde=False)


# In[29]:


sns.boxplot(x='attributes.WiFi_free', y='stars', data= select_df)


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1)
print(X_train.shape)
print(X_test.shape)


# In[31]:


lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# In[32]:


lasso.fit(X_train, y_train)
y_pred_train = lasso.predict(X_train)
print('Train RMSE:')
print(np.sqrt(mean_squared_error(y_train, y_pred_train)))

y_pred_test = lasso.predict(X_test)
print('Test RMSE:')
print(np.sqrt(mean_squared_error(y_test, y_pred_test)))


# In[33]:


ENet.fit(X_train, y_train)
y_pred_train = ENet.predict(X_train)
print('Train RMSE:')
print(np.sqrt(mean_squared_error(y_train, y_pred_train)))

y_pred_test = ENet.predict(X_test)
print('Test RMSE:')
print(np.sqrt(mean_squared_error(y_test, y_pred_test)))


# In[34]:


GBoost.fit(X_train, y_train)
y_pred_train = GBoost.predict(X_train)
print('Train RMSE:')
print(np.sqrt(mean_squared_error(y_train, y_pred_train)))

y_pred_test = GBoost.predict(X_test)
print('Test RMSE:')
print(np.sqrt(mean_squared_error(y_test, y_pred_test)))


# In[35]:


#use cross-validation instead of train-test split for a better estimation of the RMSE
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
cross_val_scores = cross_val_score(GBoost, X, y, scoring='neg_mean_squared_error', cv=kfold)
print('10-fold RMSEs:')
print([np.sqrt(-x) for x in cross_val_scores])
print('CV RMSE:')
print(np.sqrt(-np.mean(cross_val_scores))) # RMSE is the sqrt of the avg of MSEs
print('Std of CV RMSE:')
print(np.std(cross_val_scores))


# In[36]:


#parameters tuning for xgboost
def modelfit(alg,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain =xgb.DMatrix(X_train,label=y_train)
        xgtest = xgb.DMatrix(X_test)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          early_stopping_rounds=early_stopping_rounds,show_stdv=False)
        alg.set_params(n_estimators=cvresult.shape[0])#cvresult.shape[0] and alg.get_params()['n_estimators'] are same

    #Fit the algorithm on the data
    alg.fit(X_train, y_train,eval_metric='rmse')
    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    #Print model report:
    print("Score (Train): %f" % mean_squared_error(y_train.values, dtrain_predictions))
    #Predict on testing data:
    dtest_predictions = alg.predict(X_test)
    print("Score (Test): %f" % mean_squared_error(y_test.values, dtest_predictions))


# In[37]:


xgb1 = XGBRegressor(booster='gbtree',
                    objective= 'reg:linear',
                    eval_metric='rmse',
                    gamma = 0.1,
                    min_child_weight= 1.1,
                    max_depth= 5,
                    subsample= 0.8,
                    colsample_bytree= 0.8,
                    tree_method= 'exact',
                    learning_rate=0.1,
                    n_estimators=100,
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)
modelfit(xgb1)


# In[57]:


get_ipython().run_cell_magic('time', '', "\n#Grid seach for max_depth and min_child_weight tuning\n\nparam_test1 = {\n    'max_depth':[3,5,7,9],\n    'min_child_weight':[1,3,5]\n}\ngsearch1 = GridSearchCV(estimator = XGBRegressor(booster='gbtree',\n                    objective= 'reg:linear',\n                    eval_metric='rmse',\n                    gamma = 0.1,\n                    min_child_weight= 1.1,\n                    max_depth= 5,\n                    subsample= 0.8,\n                    colsample_bytree= 0.8,\n                    tree_method= 'exact',\n                    learning_rate=0.1,\n                    n_estimators=100,\n                    nthread=4,\n                    scale_pos_weight=1,\n                    seed=27),\n                    param_grid = param_test1, scoring='mean_squared_error',n_jobs=4,iid=False, cv=5)\ngsearch1.fit(X_train,y_train)")


# In[39]:


gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[58]:


get_ipython().run_cell_magic('time', '', "#try another one\nparam_test1b = {\n    'min_child_weight':[6,8,10,12]\n}\ngsearch1b = GridSearchCV(estimator = XGBRegressor(booster='gbtree',\n                    objective= 'reg:linear',\n                    eval_metric='rmse',\n                    gamma = 0.1,\n                    min_child_weight= 1.1,\n                    max_depth= 7,\n                    subsample= 0.8,\n                    colsample_bytree= 0.8,\n                    tree_method= 'exact',\n                    learning_rate=0.1,\n                    n_estimators=100,\n                    nthread=4,\n                    scale_pos_weight=1,\n                    seed=27),\n                    param_grid = param_test1b, scoring='mean_squared_error',n_jobs=4,iid=False, cv=5)\ngsearch1b.fit(X_train,y_train)")


# In[59]:


gsearch1b.grid_scores_, gsearch1b.best_params_, gsearch1b.best_score_


# In[60]:


get_ipython().run_cell_magic('time', '', "#gamma tuning\nparam_test3 = {\n    'gamma':[i/10.0 for i in range(0,5)]\n}\ngsearch3 = GridSearchCV(estimator = XGBRegressor(booster='gbtree',\n                    objective= 'reg:linear',\n                    eval_metric='rmse',\n                    gamma = 0.1,\n                    min_child_weight= 10,\n                    max_depth= 7,\n                    subsample= 0.8,\n                    colsample_bytree= 0.8,\n                    tree_method= 'exact',\n                    learning_rate=0.1,\n                    n_estimators=100,\n                    nthread=4,\n                    scale_pos_weight=1,\n                    seed=27),\n                       param_grid = param_test3, scoring='mean_squared_error',n_jobs=4,iid=False, cv=5)\ngsearch3.fit(X_train,y_train)")


# In[61]:


gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


# In[62]:


get_ipython().run_cell_magic('time', '', "#subsample and colsample_bytree tuning\nparam_test4 = {\n    'subsample':[i/10.0 for i in range(6,10)],\n    'colsample_bytree':[i/10.0 for i in range(6,10)]\n}\ngsearch4 = GridSearchCV(estimator = XGBRegressor(booster='gbtree',\n                    objective= 'reg:linear',\n                    eval_metric='rmse',\n                    gamma = 0.4,\n                    min_child_weight= 10,\n                    max_depth= 7,\n                    subsample= 0.8,\n                    colsample_bytree= 0.6,\n                    tree_method= 'exact',\n                    learning_rate=0.1,\n                    n_estimators=100,\n                    nthread=4,\n                    scale_pos_weight=1,\n                    seed=27),\n                       param_grid = param_test4, scoring='mean_squared_error',n_jobs=4,iid=False, cv=5)\ngsearch4.fit(X_train,y_train)")


# In[63]:


gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


# In[64]:


get_ipython().run_cell_magic('time', '', "#reg_alpha、reg_lambda rough tuning\n\nparam_test6 = {\n    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]\n}\ngsearch6 = GridSearchCV(estimator = XGBRegressor(booster='gbtree',\n                    objective= 'reg:linear',\n                    eval_metric='rmse',\n                    gamma = 0.4,\n                    min_child_weight= 10,\n                    max_depth= 7,\n                    subsample= 0.8,\n                    colsample_bytree= 0.6,\n                    tree_method= 'exact',\n                    learning_rate=0.1,\n                    n_estimators=100,\n                    nthread=4,\n                    scale_pos_weight=1,\n                    seed=27),\n                    param_grid = param_test6, scoring='mean_squared_error',n_jobs=4,iid=False, cv=5)\ngsearch6.fit(X_train,y_train)")


# In[65]:


gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


# In[66]:


get_ipython().run_cell_magic('time', '', "##reg_alpha、reg_lambda fine tuning\nparam_test7 = {\n    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]\n}\ngsearch7 = GridSearchCV(estimator = XGBRegressor(booster='gbtree',\n                    objective= 'reg:linear',\n                    eval_metric='rmse',\n                    gamma = 0.4,\n                    min_child_weight= 10,\n                    max_depth= 7,\n                    subsample= 0.8,\n                    colsample_bytree= 0.6,\n                    tree_method= 'exact',\n                    learning_rate=0.1,\n                    n_estimators=100,\n                    nthread=4,\n                    scale_pos_weight=1,\n                    seed=27),\n                       param_grid = param_test7, scoring='mean_squared_error',n_jobs=4,iid=False, cv=5)\ngsearch7.fit(X_train,y_train)")


# In[67]:


gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_


# In[68]:


get_ipython().run_cell_magic('time', '', "#reduce learning_rate, increase n_estimators\nparam_test9 = {\n    'n_estimators':[50, 100, 200, 500,1000],\n    'learning_rate':[0.001, 0.01, 0.05, 0.1,0.2]\n}\ngsearch9 = GridSearchCV(estimator = XGBRegressor(booster='gbtree',\n                    objective= 'reg:linear',\n                    eval_metric='rmse',\n                    gamma = 0.4,\n                    min_child_weight= 10,\n                    max_depth= 7,\n                    subsample= 0.8,\n                    colsample_bytree= 0.6,\n                    tree_method= 'exact',\n                    learning_rate=0.1,\n                    n_estimators=100,\n                    nthread=4,\n                    scale_pos_weight=1,\n                    reg_alpha=0.01,                           \n                    seed=27),\n                       param_grid = param_test9, scoring='mean_squared_error',n_jobs=4,iid=False, cv=5)\ngsearch9.fit(X_train,y_train)\n")


# In[69]:


gsearch9.grid_scores_, gsearch9.best_params_, gsearch9.best_score_


# In[52]:


xgb9 = XGBRegressor(booster='gbtree',
                    objective= 'reg:linear',
                    eval_metric='rmse',
                    gamma = 0.4,
                    min_child_weight= 10,
                    max_depth= 7,
                    subsample= 0.8,
                    colsample_bytree= 0.6,
                    tree_method= 'exact',
                    learning_rate=0.01,
                    n_estimators=500,
                    nthread=4,
                    scale_pos_weight=1,
                    reg_alpha=0.05,                           
                    seed=27)


# In[53]:


xgb9.fit(X_train,y_train)


# In[54]:


sqrt(mean_squared_error(xgb9.predict(X_test),y_test))


# In[55]:


fig, ax = plt.subplots(1, 1, figsize=(8, 13))
xgb.plot_importance(xgb9, max_num_features=20, height=0.5, ax=ax)

