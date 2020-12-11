#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
df=pd.read_csv(r'C:\Users\admin\Desktop\Course\DataSet\nrippner-titanic-disaster-dataset\titanic.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.isnull()


# In[6]:


sns.heatmap(df.isnull(),yticklabels=False,xticklabels=True,cbar=False,cmap='viridis')


# In[7]:


sns.countplot(x='survived',data=df)


# In[8]:


sns.countplot(x= 'survived',hue='sex',data=df)


# In[9]:


sns.countplot(x= 'survived',hue='pclass',data=df)


# In[10]:


sns.distplot(df['age'].dropna(),kde=False,bins=20)


# In[11]:


#Distplot for those who didnot survived
#df1=df[['age','survived'].df['survived']==0]
#df1
#sns.distplot(df[df['age'].df['survived']==0].dropna(),kde=False,bins=20)


# In[12]:


df['age'].hist(bins=30,alpha=0.3)


# In[13]:


sns.countplot(x='sibsp',data=df)


# In[14]:


df['fare'].hist(bins=40,figsize=(8,4))


# In[15]:


plt.figure(figsize=(8,4))
sns.boxplot(x='pclass',y='age',data=df)


# In[16]:


def compute_age(cols):
    age=cols[0]
    pclass=cols[1]
    
    if pd.isnull(age):
        
        if pclass == 1:
            return 37
        
        elif pclass == 2:
            return 29
        
        else:
            return 24
        
    else:
        return age


# In[17]:


df['age']=df[['age','pclass']].apply(compute_age, axis=1)


# In[18]:


sns.heatmap(df.isnull(),yticklabels=False,xticklabels=True,cbar=False,cmap='viridis')


# In[19]:


df.drop(['cabin','boat','body','home.dest'],axis=1,inplace=True)


# In[20]:


sns.heatmap(df.isnull(),yticklabels=False,xticklabels=True,cbar=False,cmap='viridis')


# In[21]:


df.head()


# In[22]:


df.info()


# In[23]:


pd.get_dummies(df['embarked'],drop_first=True,).head()


# In[24]:


pd.get_dummies(df['sex'],drop_first=True).head()


# In[25]:


sex = pd.get_dummies(df['sex'],drop_first=True)
embark = pd.get_dummies(df['embarked'],drop_first=True,)


# In[26]:


df.drop(['sex','embarked','name','ticket'],axis=1,inplace=True)


# In[27]:


df.head()


# In[28]:


df = pd.concat([df,sex,embark], axis=1)


# In[29]:


df.head()


# In[30]:


df.dropna(inplace=True)


# In[31]:


df.isnull().sum()


# In[32]:


train=df.drop('survived',axis=1)


# In[33]:


train.head()


# In[34]:


test=df['survived']


# In[35]:


test.head()


# In[36]:


# Logistic Regression
from  sklearn.model_selection import train_test_split
from  sklearn.linear_model import LogisticRegression


# In[37]:


X_train,X_test,y_train,y_test = train_test_split(train,test,test_size=0.3,random_state=101)


# In[38]:


logit_model = LogisticRegression()
logit_model.fit(X_train,y_train)


# In[39]:


predictions = logit_model.predict(X_test)


# In[40]:


from sklearn.metrics import confusion_matrix


# In[41]:


accuracy = confusion_matrix(y_test,predictions)


# In[42]:


accuracy


# In[43]:


from sklearn.metrics import accuracy_score


# In[44]:


accuracy = accuracy_score(y_test,predictions)
accuracy


# In[ ]:




