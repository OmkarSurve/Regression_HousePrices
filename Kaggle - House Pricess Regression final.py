
# coding: utf-8

# Necessar Library imports

# In[133]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')


# Data Import and Overview

# In[135]:


df_orig = pd.read_csv('Data/train.csv',index_col='Id')


# Combining Train and Test dataset for data explration and feature selection 

# In[136]:


df = pd.read_csv('Data/train.csv',index_col='Id')
pd.options.display.max_columns = None


# In[137]:


dft = pd.read_csv('Data/test.csv',index_col='Id')
pd.options.display.max_columns = None


# In[138]:


df = pd.concat([df,dft])


# Let us do some initial data analysis to know more about the data

# In[139]:


df.info()


# In[140]:


df.describe()


# In[141]:


df.head()


# Let us do some data exploration to see how few features relate to Final Sale Price

# Let us compare how number of bedroom corelates with Final House Prices

# In[143]:


sns.set_style('whitegrid')
sns.jointplot(x=df['BedroomAbvGr'],y=df['SalePrice'])


# Let us compare how overall quality corelates with Final House Prices

# In[144]:


sns.jointplot(x=df['OverallQual'],y=df['SalePrice'])


# We can see clearly, that as the Quality increases so the does the prices

# Now, lets check few numerical columns and see the relationship. Let us see if Basement area has any impact on Sale Price

# In[145]:


sns.jointplot(x=df['TotalBsmtSF'],y=df['SalePrice'])


# We can see it has somewhat linear relation ship with few outliers. Lets compare Living Room area.

# In[146]:


sns.jointplot(x=df['GrLivArea'],y=df['SalePrice'])


# Somewhat similiar relationship as Basement area.

# Now lets us do some feature selection to check if we need all the columns

# ##### Feature Engineering ###########

# Data Analysis to check if a certain columns contain, almost all records having a single value. This columns can be removed as they wont contribute to the final amount. We are doing this only on categorical variables

# In[148]:


not_imp_columns = [columns for columns in df.select_dtypes(include='object').columns if df[columns].value_counts()[0] >= 0.8 * 1460]
not_imp_columns


# We should drop this columns as they are not useful in predicting Final Price due to low variance

# In[149]:


df.drop(not_imp_columns,axis=1,inplace=True)


# Similiarly, let us plot a heatmap to check correlation among different predictor varables and the Sale Price. We need to remove variables if they are highly correlated with each other and keep only 1 amonng those variables as keeping both the variables reduces the model accuracy. For simiplification storing the int columns in the array.

# In[150]:


int_corr = ['LotFrontage',
 'LotArea',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'LowQualFinSF',
 'GrLivArea',
 'GarageArea',
 'WoodDeckSF',
 'OpenPorchSF',
 'EnclosedPorch',
 '3SsnPorch',
 'ScreenPorch',
 'PoolArea',
 'MiscVal',
 'SalePrice']


# In[151]:


fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(df[int_corr].corr(),annot=True,linecolor='white',lw=0.1)


# Based on correlation matrix above, Removing Column TotalBsmtSF as it is highly correlated with column 1stFlrSF

# In[152]:


df.drop('TotalBsmtSF',axis=1,inplace=True)


# Based on correlation matrix above, Removing below Columns as they have very low correlation with target column Sale Price

# In[153]:


df.drop(['BsmtFinSF2','LowQualFinSF', 'MiscVal', 'ScreenPorch', '3SsnPorch', 'EnclosedPorch', 'BsmtUnfSF','PoolArea'],
        axis=1,inplace=True)


# Let us check, how many columns remain in our dataset after initial feature selection

# In[171]:


df.info()


# Now let us focus on missing values. We need to impute the missing values for the columns before feeding it into the model.

# Removing below columns as they have lot of null values

# In[156]:


df.drop(['MiscFeature','PoolQC','Alley', 'Fence'],axis=1,inplace=True)


# Lets focus on filling the null values

# In[157]:


df['LotFrontage'] = df['LotFrontage'].fillna(value=df['LotFrontage'].mean())


# In[159]:


df['MasVnrArea'] = df['MasVnrArea'].fillna(0)


# In[161]:


df['BsmtFinType1'] = df['BsmtFinType1'].fillna(value='No Basement')


# In[162]:


df['FireplaceQu'] = df['FireplaceQu'].fillna(value='No Fireplace')


# For the Garage Built Year, we have few missing values. Let us see the distribution of the column first.

# In[163]:


df['GarageYrBlt'].hist()


# We need to fill the missing values with some year which is not part of the distribution, so that our model uniquely identifies such records

# In[164]:


df['GarageYrBlt'] = df['GarageYrBlt'].fillna(1800)


# In[168]:


df['BsmtFinSF1'] = df['BsmtFinSF1'].fillna(0)
df['BsmtFullBath'] = df['BsmtFullBath'].fillna(0)
df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)
df['GarageCars'] = df['GarageCars'].fillna(0)
df['GarageArea'] = df['GarageArea'].fillna(0)


# Couple of Categorical columns, have just 1 missing value. We will replace them with most common value of that column

# In[170]:


df[['Exterior1st','Exterior2nd']] = df[['Exterior1st','Exterior2nd']].apply(lambda x:x.fillna(x.value_counts().index[0]))


# We have removed few columns using filter method. We will now built model using the remaining columns

# Let us encode the Categorical values so that can be used in the Model. We will encode all Nominal Categorical values using one hot encoder.

# MSSubClass is defined as integer, but is actually a categorical variable, so we convert it to Categorical

# In[174]:


df['MSSubClass'] = df['MSSubClass'].astype(int).astype(object)


# In[175]:


df = pd.get_dummies(df, columns=['MSSubClass','Neighborhood','Exterior1st', 'Exterior2nd'
          ], prefix=['MSSubClass','Neighborhood','Exterior1st', 'Exterior2nd'
          ])


# Now let us encode ordinal columns

# In[177]:


df['BsmtFinType1'].value_counts()


# In[178]:


BsmtFinType1_dict = {
    'No Basement':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4, 'ALQ':5, 'GLQ':6
}


# In[179]:


df['BsmtFinType1']=df['BsmtFinType1'].map(BsmtFinType1_dict)


# In[180]:


df['FireplaceQu'].value_counts()


# In[181]:


FireplaceQu_dict = {
    'No Fireplace':0,'Po':1,'Fa':2,'TA':3,'Gd':4, 'Ex':5
}


# In[182]:


df['FireplaceQu']=df['FireplaceQu'].map(FireplaceQu_dict)


# We are done with Data Cleaning and Feature Selection. Now let us apply few models and compare the results. We will first divide the data into training and testing. As per the Kaggle round, we have half the data as training and half as testing.

# In[186]:


X = df.drop('SalePrice',axis=1)
y = df['SalePrice']


# In[187]:


from sklearn.model_selection import train_test_split


# In[188]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)


# Let us apply few models and compare the results. We will start with Linear Regression.

# In[189]:


from sklearn.linear_model import LinearRegression, LogisticRegression


# In[190]:


lm = LinearRegression()


# In[191]:


lm.fit(X_train,y_train)


# In[193]:


linear_pred = lm.predict(X_test)


# In[194]:


from sklearn import metrics


# Root Mean Square to evaluate model. RMSE should be low.

# In[196]:


np.sqrt(metrics.mean_squared_error(y_test,linear_pred))


# We can also plot actual vs predicted value to evaluate the model. The more linear the correlation better the model.

# In[201]:


plt.scatter(y_test,predictions)


# Let us apply Logistic Regression. Logistic Regression is mostly used for Classification problems, but we can try to build a regression model as well.

# In[197]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[198]:


pred_log = logmodel.predict(X_test)


# In[199]:


np.sqrt(metrics.mean_squared_error(y_test,pred_log))


# In[200]:


plt.scatter(y_test,predictions)


# Let us try Lasso Model

# In[202]:


from sklearn import linear_model


# In[203]:


clf = linear_model.Lasso(alpha=0.1,normalize=True)


# In[204]:


clf.fit(X_train,y_train)


# In[205]:


lass_pred = clf.predict(X_test)


# In[207]:


np.sqrt(metrics.mean_squared_error(y_test,lass_pred))


# SVR Linear Kernel

# In[208]:


from sklearn.svm import SVR


# In[209]:


svr_lin = SVR(kernel='linear', gamma='auto', C=1.0)
svr_lin.fit(X_train, y_train)


# In[210]:


svr_pred = svr_lin.predict(X_test)


# In[211]:


np.sqrt(metrics.mean_squared_error(y_test,svr_pred))


# Gradient Boost Model

# In[212]:


from sklearn import ensemble


# In[213]:


clf = ensemble.GradientBoostingRegressor(n_estimators=400,max_depth=5,min_samples_split=2,learning_rate=0.1,loss='ls')


# In[214]:


clf.fit(X_train,y_train)


# In[216]:


predict_gb = clf.predict(X_test)


# In[217]:


np.sqrt(metrics.mean_squared_error(y_test,predict_gb))


# WE found out that, Gradient Boost model gave the best results of all the model. As mentioned above, we can train our model on the training the dataset and evaluate on the test dataset. RMSE is a good measure the evaluate the model. Kaggle requires us to upload the predicted value to the website to get the score, which tells us how our model fared. For testing purpose we had divided the training data into train and test so that we can check how the model is performing.
