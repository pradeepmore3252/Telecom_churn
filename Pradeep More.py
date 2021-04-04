#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("hello world")


# In[2]:


import numpy as np


# In[3]:


import pandas as pd


# In[4]:


import seaborn as sns


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


import matplotlib.ticker as mtick


# In[7]:


telecom_cust_churn = pd.read_csv('F:/PRADEEP/CASE STUDY PYTHON/Churn.csv')


# In[8]:


telecom_cust_churn


# In[39]:


telecom_cust_churn.head()


# In[40]:


telecom_cust_churn.columns.values


# In[14]:


telecom_cust_churn.head()


# In[15]:


telecom_cust_churn.columns.values


# In[16]:


# Checking the data types of all the columns
telecom_cust_churn.dtypes


# In[19]:


# Converting Total Charges to a numerical data type.
telecom_cust_churn.TotalCharges = pd.to_numeric(telecom_cust_churn.TotalCharges, errors='coerce')
telecom_cust_churn.isnull().sum()


# In[20]:


#Removing missing values 
telecom_cust_churn.dropna(inplace = True)
#Remove customer IDs from the data set
df2 = telecom_cust_churn.iloc[:,1:]
#Converting the predictor variable in a binary numeric variable
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

#Let's convert all the categorical variables into dummy variables
df_dummies = pd.get_dummies(df2)
df_dummies.head()


# In[21]:


#Get Correlation of "Churn" with other variables:
plt.figure(figsize=(15,8))
df_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# In[ ]:


# Observation
Month to month contracts, absence of online security and tech support seem to be positively correlated with churn. While, tenure, two year contracts seem to be negatively correlated with churn.

Interestingly, services such as Online security, streaming TV, online backup, tech support, etc. without internet connection seem to be negatively related to churn.

We will explore the patterns for the above correlations below before we delve into modelling and identifying the important variables.


# In[22]:


# Relation Between Monthly and Total Charges
telecom_cust_churn[['MonthlyCharges', 'TotalCharges']].plot.scatter(x = 'MonthlyCharges',
                                                              y='TotalCharges')


# In[23]:


sns.boxplot(x = telecom_cust_churn.Churn, y = telecom_cust_churn.tenure)


# In[24]:


#1 Logistic Regression
# We will use the data frame where we had created dummy variables
y = df_dummies['Churn'].values
X = df_dummies.drop(columns = ['Churn'])

# Scaling all the variables to a range of 0 to 1
from sklearn.preprocessing import MinMaxScaler
features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features


# In[25]:


# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[26]:


# Running logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)


# In[27]:


from sklearn import metrics
prediction_test = model.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(y_test, prediction_test))


# In[28]:


# To get the weights of all the variables
weights = pd.Series(model.coef_[0],
                 index=X.columns.values)
print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))


# In[30]:


print(weights.sort_values(ascending = False)[-10:].plot(kind='bar'))


# In[ ]:


# Observations

We can see that some variables have a negative relation to our predicted variable (Churn), while some have positive relation. Negative relation means that likeliness of churn decreases with that variable. Let us summarize some of the interesting features below:

1) As we saw in our EDA, having a 2 month contract reduces chances of churn. 2 month contract along with tenure have the most negative relation with Churn as predicted by logistic regressions
2) Having DSL internet service also reduces the proability of Churn
3) Lastly, total charges, monthly contracts, fibre optic internet services and seniority can lead to higher churn rates. This is interesting because although fibre optic services are faster, customers are likely to churn because of it. I think we need to explore more to better understad why this is happening.
Any hypothesis on the above would be really helpful!


# In[31]:


#2 Random Forest
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))


# In[32]:


importances = model_rf.feature_importances_
weights = pd.Series(importances,
                 index=X.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')


# In[ ]:


# Observations:

1) From random forest algorithm, monthly contract, tenure and total charges are the most important predictor variables to predict churn.
2) The results from random forest are very similar to that of the logistic regression and in line to what we had expected from our EDA


# In[33]:


#3 Support Vector Machine (SVM)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)


# In[35]:


from sklearn.svm import SVC

model.svm = SVC(kernel='linear') 
model.svm.fit(X_train,y_train)
preds = model.svm.predict(X_test)
metrics.accuracy_score(y_test, preds)


# In[36]:


# Create the Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,preds))


# In[ ]:


# Observation
Wth SVM I was able to increase the accuracy to upto 82%.

