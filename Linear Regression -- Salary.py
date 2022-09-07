#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Importing required libariries
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dt =pd.read_excel(r"C:\Users\admin\Desktop\Course\EXL Course/Salary.xlsx")
dt.isnull().sum()


# In[4]:


dt.plot(x='YearsExperience',y='Salary',style='*', color= 'green')
#d.plot(x='Study hours', y='Scores', style='*') 
plt.title('Years of Exp## Importing required libariries
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', "inlineerience Vs Salary')")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[5]:


#Check NULL valuedt.isnull().sum()
dt.isnull().sum()


# In[6]:


dt.describe()


# In[7]:


import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set()
plt.figure(figsize=(5,5))
cor = dt.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[8]:


X = dt.iloc[:, :-1].values  ## Feature
y = dt.iloc[:, 1].values    ## Target


# In[9]:


#Scikit-Learn's built-in train_test_split() method
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# In[10]:


#Training the Algorithm
from sklearn.linear_model import LinearRegression  
reg = LinearRegression()  
reg.fit(X_train, y_train) 


# In[11]:


#value for the intercept and slope
print(reg.intercept_) 


# In[12]:


print(reg.coef_)  


# In[13]:


#For every one unit of change in years of experience, the change in the score is about 5354 salary increase.


# In[14]:


y_pred = reg.predict(X_test)  
y_pred 


# In[15]:


r = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
r 


# In[16]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  #average of absolute errors
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 


# In[17]:


#Performance Improvement by Cross validation
from sklearn.model_selection import train_test_split  
train, validation = train_test_split(dt, test_size=0.50, random_state = 5)


# In[18]:


X_train, X_v, y_train, y_v = train_test_split(X, y, test_size=0.5, random_state=5) 
from sklearn.linear_model import LinearRegression  
reg = LinearRegression()  
reg.fit(X_train, y_train) 


# In[19]:


print(reg.intercept_,reg.coef_)


# In[20]:


y_pred = reg.predict(X_v)  
y_pred


# In[21]:


from sklearn.metrics import mean_squared_error, r2_score
r2 = r2_score(y_v, y_pred)
print(r2)


# In[22]:


from sklearn.metrics import mean_squared_error, r2_score
r2 = r2_score(y_v, y_pred)
print(r2)


# In[23]:


from sklearn import metrics 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_v, y_pred)))


# In[24]:


#Model Correction--k-fold cross-validation
from sklearn.model_selection import cross_val_score
lm = LinearRegression()
scores = cross_val_score(lm, X_train, y_train, scoring='r2', cv=5)
scores  


# In[25]:


# can tune other metrics, such as MSE
scores = cross_val_score(lm, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
scores


# In[ ]:





# In[ ]:




