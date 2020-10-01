#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats  as stats
import matplotlib.pyplot as plt



# In[2]:


df= pd.read_csv(r'winequality.csv')


# In[3]:


df.columns=['type', 'fixedacidity', 'volatileacidity', 'citricacid',
       'residualsugar', 'chlorides', 'freesulfurdioxide',
       'totalsulfurdioxide', 'density', 'pH', 'sulphates', 'alcohol',
       'quality']


# In[4]:


df.dtypes


# In[5]:


### finding the misssing values
na_column = [col for col in df.columns if df[col].isnull().sum()>1]
na_column
for col in na_column:
    print(col, np.round(df[col].isnull().mean(), 4 ), ' % missing values')


# In[6]:


df.isnull().sum()


# In[7]:


for col in na_column:
    data=df.copy()
    data[col]= np.where(df[col].isnull(),1,0)
    data.groupby(col).quality.median().plot.bar()
    plt.show()
    plt.xlabel('col')
    plt.ylabel('quality')


# #### Here With the relation between the missing values and the dependent variable is clearly visible.So We need to replace these nan values with something meaningful which we will do in the Feature Engineering section

# ### here  we will find all the numerical columns from the datset

# In[8]:


numerical_col=[col for col in df.columns if df[col].dtype != 'O']
print('total numercal columns are :-', len(numerical_col)     )
for col in numerical_col:
    print('the all numrical columns names are :-', col )
    


# In[9]:


# here we are trying to analyse that how much no of unique values dataframe have
df.nunique()


# ### here we are trying to look that how many columns are  contineous in the data frame and what behaviour they follow with respect to the quality

# In[10]:


contineous_col=[col for col in df.columns if len(df[col].unique())> 25]
print('total numercal columns are :-', len(contineous_col)     )
for col in contineous_col:
    print('the all contineous columns names are :-', col )
    
    
   


# In[11]:


df[df['fixedacidity'].isnull()]==True


# In[12]:


df.isnull().sum()


# #### here we are trying to replace fixed acidity nan value with the mean value

# In[13]:


df.groupby('type').mean()['fixedacidity']


# In[14]:


#what we are trying to do here is that we are replacing all the fixed acidity columns emptied at red wine with the above mean value
df.loc[(df['type']=='red') & (df['fixedacidity'].isnull()==True),'fixedacidity']=8.322104


# In[15]:


df.loc[(df['type']=='white') & (df['fixedacidity'].isnull()==True),'fixedacidity']=6.855532


# #### here we are trying to replace volatile acidity nan value with the mean value

# In[16]:


df.groupby('type').mean()['volatileacidity']


# In[17]:


df.loc[(df['type']=='red') & (df['volatileacidity'].isnull()==True),'volatileacidity']=0.527738


# In[18]:


df.loc[(df['type']=='white') & (df['volatileacidity'].isnull()==True),'volatileacidity']=0.278252


# #### here we are trying to replace citric acid nan value with the mean value

# In[19]:


df.groupby('type').mean()['citricacid']


# In[20]:


df.loc[(df['type']=='red') & (df['citricacid'].isnull()==True),'citricacid']=0.271145


# In[21]:


df.loc[(df['type']=='white') & (df['citricacid'].isnull()==True),'citricacid']=0.334250


# #### here we are trying to replace residual sugar nan value with the mean value

# In[22]:


df.groupby('type').mean()['residualsugar']


# In[23]:


df.loc[(df['type']=='white') & (df['residualsugar'].isnull()==True),'residualsugar']=0.334250


# #### here we are trying to replace chlorides nan value with the mean value

# In[24]:


df.groupby('type').mean()['chlorides']


# In[25]:


df.loc[(df['type']=='white') & (df['chlorides'].isnull()==True),'chlorides']=0.045778


# #### here we are trying to replace ph nan value with the mean value

# In[26]:


df.groupby('type').mean()['pH']


# In[27]:


df.loc[(df['type']=='white') & (df['pH'].isnull()==True),'pH']=3.188203


# In[28]:


df.loc[(df['type']=='red') & (df['pH'].isnull()==True),'pH']=3.310864


# #### here we are trying to replace sulphate nan value with the mean value

# In[29]:


df.groupby('type').mean()['sulphates']


# In[30]:


df.loc[(df['type']=='white') & (df['sulphates'].isnull()==True),'sulphates']=0.489835


# In[31]:


df.loc[(df['type']=='red') & (df['sulphates'].isnull()==True),'sulphates']=0.658078


# In[ ]:





# In[32]:


df.isnull().sum()


# In[33]:


for col in contineous_col:
    data=df.copy()
      
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    data[col].hist(bins=25)
    plt.xlabel(col)
    plt.ylabel('count')
    
    plt.subplot(1, 2, 2)
    stats.probplot(data[col], dist="norm", plot=plt)
    plt.show()
    print(data[col].skew())
    
    
    # here as you can see the skewness of different contineous problems so that you can  see the probability graph


# ### as we can see that the most of the catagorical values are little skewed towards right (+skewed) so we will try to normal them with the help of some of the transformations given below

# ### different types of transformation

# In[34]:


g =['volatileacidity','citricacid', 'residualsugar','chlorides','sulphates']
for col in g:
   data=df.copy()
   if data[col].name=='total sulfur dioxide':#here we are defining the col name if we do not decalre name if will 
                                              # throw a ambiguity erro
       pass
   else:
       data[col]=np.log(data[col])
       
       plt.figure(figsize=(15,7))
       plt.subplot(1,2,1)
       plt.scatter(data[col],data['quality'])
       plt.xlabel(col)
       plt.ylabel('quality')
      
       plt.subplot(1,2,2)
       stats.probplot(data[col], dist="norm", plot=plt)
       plt.xlabel(col)
       plt.show()
       print('the skewness of the', col ,'after the log transformationo is', data[col].skew())
       


# In[35]:


g =['volatileacidity','residualsugar','chlorides','sulphates']
for col in g:
    data=df.copy()
    if data[col].name=='total sulfur dioxide':#here we are defining the col name if we do not decalre name if will 
                                               # throw a ambiguity erro
        pass
    else:
        df[col]=np.log(data[col])


# In[36]:


""" in the above coloumn we are tranforming only those columns which have a imorved transformation after the log transformation 
if we compare the above two graphical columns we can easily see that we have clearly a better tansformation for 
['volatile acidity','residual sugar','chlorides','sulphates'] these columns so what we have done is that saving that log transformation 
in the orginal column """


# In[37]:


""" here in the below coloumn we are trying to see a one more tansformation which is sqrt transformation but
 we have seen in the above column is that the log transformation is better than the sqrt transformation because the vlaues for 
 the (=['citric acid', 'residual sugar']) are more colser to zero and hence makes it normal skewed
 if it gives a +ve values it becomes right skewed 
 and if it gives a -ve value it becomes the left skewed 
 so in this condition we will only take the log transformation"""
# but below column is only for demosntration


# In[38]:


h =['citricacid', 'residualsugar']
for col in h:
   data=df.copy()
   if data[col].name=='total sulfur dioxide':#here we are defining the col name if we do not decalre name if will 
                                              # throw a ambiguity erro
       pass
   else:
       data[col]=np.sqrt(data[col])
       
       plt.figure(figsize=(15,7))
       plt.subplot(1,2,1)
       plt.scatter(data[col],data['quality'])
       plt.xlabel(col)
       plt.ylabel('quality')
      
       plt.subplot(1,2,2)
       stats.probplot(data[col], dist="norm", plot=plt)
       plt.xlabel(col)
       plt.show()
       print('the skewness of the', col ,'after the log transformationo is', data[col].skew())
       


# ### here we are trying to see the outliesrs in the contineouse variables

# In[39]:


for col in contineous_col:
    data =df.copy()
    data.boxplot(column=col)
    plt.ylabel(col)
    plt.show()


# ### now here we see that how many variables are catagorical in the dataset

# In[40]:


catagorical_col = [col for col in df.columns if df[col].dtype == 'O']
print(catagorical_col)


# ### so this means that there is  only one catagorical column

# ###  Now from here the part of the feature engineering starts  

# In[41]:


df


# In[42]:


df.isnull().sum()


# #### what we are doing in the below code is that we are creating a new columns  so that it can be known that in which rows we have filled values although this dataset doesnt have any nan values but the below code is only for the demonstration that how to fill the nan values with median

# In[43]:


for col in contineous_col:
    median=df[col].median()
    print(col,':---',median)
    
    df[col+'_nan']=np.where(df[col].isnull(),1,0)  # what we are doing here is that we are creating a new catagory
    df[col].fillna(median,inplace=True)            # so that it can be known that in which rows we have filled values
    


# In[44]:


df 


# In[45]:


df.isnull().sum()


# ### now as you can see there is no empty numerical col in the dataset remains

# In[46]:


for col in catagorical_col:
    temp=df.groupby(col)['quality'].count()/len(df)
    print(temp)
    temp_index=temp[temp < 0.01].index
    df[col]=np.where(df[col].isin(temp_index),'rare_var',df[col])


# #####  the above code is super important so please have a look in it and at the last line of code we are trying to match the results by applying isin function

# In[47]:


df


# ### now we have to do the  log normalization for the numerical columns

# In[48]:


df


# ### before giving the dataset to the maching learning we have to make the dummies for the catagorical variables 

# In[49]:


for col in catagorical_col:
    df[col]=pd.get_dummies(df[col])


# In[50]:


df


# In[ ]:





# In[51]:


#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2,5, 9)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)


# In[52]:


df['quality']=pd.get_dummies(df['quality'])


# In[53]:


df.groupby('quality').count()


# In[54]:


dd=df.copy() # we are making copy because we are going to use it in the later part 


# In[ ]:





# #### now as a part of Feature selection first we have to do the scaling of the variables

# In[ ]:





# In[55]:


#independent_col= [col for col in df.columns if col not in df['quality']]
independent_col= ['type','fixedacidity','volatileacidity','citricacid','residualsugar','chlorides','freesulfurdioxide','totalsulfurdioxide','density','pH']
independent_col


# In[56]:


from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(df[independent_col])


# In[57]:


scaler.transform(df[independent_col])


# In[58]:


df.columns


# In[59]:


df.drop(['fixedacidity_nan','volatileacidity_nan', 'citricacid_nan', 'residualsugar_nan','chlorides_nan', 'freesulfurdioxide_nan', 'totalsulfurdioxide_nan','density_nan', 'pH_nan', 'sulphates_nan', 'alcohol_nan'],axis=1,inplace=True)


# In[60]:


df


# In[61]:


df.to_csv('wine_quality.csv',index=False)


# #### this is not the part of above code but this is just to show you that there is a another way of doing feature scaling and that is  Z Score(Standardization)  not just min max scaler
# 

# In[62]:


dd


# In[63]:


for col in contineous_col:
    mean_list=dd[col].mean()
    print(mean_list)  # this is the mean list for each of the catagorical coloum


# In[64]:


for col in contineous_col:
    mean_list=dd[col].std()
    print(mean_list)  # this is the standard deviation list for each of the catagorical coloum


# In[65]:


## so we know that there is a formula for calculating the 
## z score is subtracting the value by the mean and dividing it by the standard deviation.
## We can either do it by using the values of mean and standard deviation calculated above 
## or we can use preprocessing.scale for doing this.


# In[66]:


from sklearn import preprocessing


# In[67]:


list=[]
for col in contineous_col:
    list.append(col)
    
    
dd=preprocessing.scale(dd[list])


# In[68]:


dd


# In[69]:


# but usually we do the standard scaler of min max scaler scaling and not the above one this is for demonstration
# we again going to use the df dataset


# ### from here the part of the machine learning starts with some part of feature selection 

# In[70]:


df=pd.read_csv(r'wine_quality.csv')
df


# In[ ]:


# if there are number of columns in the dataset if we want to see all the columns while reading the dataset 
# we have to write some command
pd.pandas.set_option('display.max_columns',None)


# In[71]:


X=df.copy()
X.drop('quality',axis=1,inplace=True)
Y=df['quality']


# In[72]:


X


# In[73]:


Y


# In[74]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=11)


# In[75]:


X_train.shape , X_test.shape


# ####  this is the first model technique we are trying to see in this scenario

# In[76]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[77]:


lg = LogisticRegression()


# In[78]:


lg.fit(X_train, Y_train)


# In[79]:


Y_pred =lg.predict(X_test)


# In[80]:


accuracy = metrics.accuracy_score(Y_test,Y_pred)
accuracy_percen= accuracy*100
accuracy_percen


# In[ ]:





# #### here we are applying the second alogrithm Random forest classifier

# In[81]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,Y_train)


# In[82]:


y_pred =classifier.predict(X_test)


# In[83]:


from sklearn.metrics import accuracy_score


# In[84]:


accuracy1= accuracy_score(Y_test,y_pred)


# In[85]:


accuracy1


# In[ ]:





# In[86]:


import pickle
pickle_out=open('classifier.pkl','wb')
pickle.dump(classifier,pickle_out)
pickle_out.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[37]:





# In[ ]:





# In[ ]:




