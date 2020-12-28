#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Download all important libraries.


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# Import data


# In[4]:


df = pd.read_excel(r'C:\Users\admin\Desktop\Course\DataSet\Flight Fare Prediction\Data_Train.xlsx')


# In[5]:


len(df)


# In[6]:


df.head()


# In[11]:


df.columns


# In[16]:


df.shape


# In[12]:


pd.set_option('display.max_columns', None)


# In[13]:


df.head()


# In[14]:


df.info()


# Note: Here all columns are object data type except price that is integer. As we are predicting price of airline tickets, here price is dependent feature (y) and all other features are independent features (x)

# In[17]:


#Removing Null value 


# In[18]:


df.isnull().sum()


# In[19]:


df.dropna(inplace=True)


# In[20]:


df.isnull().sum()


# Exploratory Data Anlysis 

# In[24]:


df["Journey_day"] = pd.to_datetime(df.Date_of_Journey, format="%d/%m/%Y").dt.day


# In[25]:


df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"], format = "%d/%m/%Y").dt.month


# In[26]:


df.head()


# In[27]:


df.drop(["Date_of_Journey"],axis=1,inplace=True)


# In[28]:


df.columns


# In[30]:


# Extract time as hours and minute and last drop that column.
df["Dep_hour"]=pd.to_datetime(df["Dep_Time"]).dt.hour
df["Dep_min"]=pd.to_datetime(df["Dep_Time"]).dt.minute

#Drop the column Dep_Time
df.drop(["Dep_Time"], axis = 1,inplace=True)


# In[31]:


df.columns


# In[32]:


df.head()


# In[33]:


df["Arrival_hour"] = pd.to_datetime(df.Arrival_Time).dt.hour
df["Arrival_min"] = pd.to_datetime(df.Arrival_Time).dt.minute


# In[34]:


df.columns


# In[36]:


df.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[37]:


df.columns


# In[38]:


df.head()


# In[39]:


duration = list(df["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[40]:


df["Duration_hours"] = duration_hours
df["Duration_mins"]  = duration_mins


# In[41]:


df.head()


# In[43]:


df.columns


# In[52]:


df.drop(["Duration"], axis = 1, inplace = True)


# In[53]:


df.columns


# In[54]:


df.head()


# Handling Categorical Features

# In[56]:


df['Airline'].value_counts()


# In[79]:


sns.set(style='whitegrid')
sns.catplot(x="Airline", y="Price", data=df,height=10);


# In[80]:


sns.catplot(y = "Price", x = "Airline", data = df.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)


# In[83]:


sns.catplot(y = "Price", x = "Source", data = df.sort_values("Price", ascending = False), kind="boxen", height = 4, aspect = 3)


# In[86]:


sns.catplot(y = "Price", x = "Destination", data = df.sort_values("Price", ascending = False), kind="boxen", height = 4, aspect = 3)


# In[81]:


# As Airline is Nominal Categorical data we will perform OneHotEncoding

Airline = df[["Airline"]]

Airline = pd.get_dummies(Airline, drop_first= True)

Airline.head()


# In[82]:


df["Source"].value_counts()


# In[85]:


# As Source is Nominal Categorical data we will perform OneHotEncoding

Source = df[["Source"]]

Source = pd.get_dummies(Source, drop_first= True)

Source.head()


# In[88]:


df["Destination"].value_counts()


# In[90]:


# As Destination is Nominal Categorical data we will perform OneHotEncoding

Destination = df[["Destination"]]

Destination = pd.get_dummies(Destination, drop_first = True)

Destination.head()


# In[93]:


df["Route"].value_counts()


# In[94]:


df.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


# In[95]:


df.head()


# In[96]:


df["Total_Stops"].value_counts()


# In[97]:


df.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


# In[98]:


df.head()


# In[106]:


# Price vs Stops
sns.set(color_codes=True)
sns.lmplot(x="Total_Stops", y="Price", data=df);


# In[107]:


sns.scatterplot(data=df,x="Total_Stops", y="Price")


# In[102]:


df.Total_Stops.value_counts()


# In[109]:


df1 = pd.concat([df, Airline, Source, Destination], axis = 1)


# In[110]:


df1.head()


# In[111]:


df1.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)


# In[112]:


df1.head()


# In[114]:


df1.shape


# In[116]:


train_data=df1.copy()


# In[117]:


train_data.head()


# In[115]:


# Processing test data


# In[118]:


test_data = pd.read_excel(r"C:\Users\admin\Desktop\Course\DataSet\Flight Fare Prediction\Test_set.xlsx")
test_data.head()


# In[119]:


# Preprocessing

print("Test data Info")
print("-"*75)
print(test_data.info())

print()
print()

print("Null values :")
print("-"*75)
test_data.dropna(inplace = True)
print(test_data.isnull().sum())

# EDA

# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)


# Categorical data

print("Airline")
print("-"*75)
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)

print()

print("Source")
print("-"*75)
print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first= True)

print()

print("Destination")
print("-"*75)
print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)

data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

print()
print()

print("Shape of test data : ", data_test.shape)


# In[120]:


data_test.head()


# In[121]:


# Feature Selection


# In[123]:


train_data.columns


# In[124]:



X = train_data.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()


# In[125]:


y = train_data.iloc[:, 1]
y.head()


# In[127]:


plt.figure(figsize = (20,20))
sns.heatmap(train_data.corr(), annot = True, cmap = "RdYlGn")

plt.show()


# In[128]:


from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)


# In[129]:


print(selection.feature_importances_)


# In[130]:


plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# Random Forest

# In[131]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[132]:


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)


# In[133]:


y_pred = reg_rf.predict(X_test)


# In[134]:


reg_rf.score(X_train, y_train)


# In[135]:


reg_rf.score(X_test, y_test)


# In[136]:


sns.distplot(y_test-y_pred)
plt.show()


# In[137]:


plt.scatter(y_test, y_pred, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[138]:


from sklearn import metrics


# In[139]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[140]:


# RMSE/(max(DV)-min(DV))

2090.5509/(max(y)-min(y))


# In[141]:


metrics.r2_score(y_test, y_pred)


# Hyperparameter Tuning

# In[142]:


from sklearn.model_selection import RandomizedSearchCV

#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# In[143]:


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}


# In[144]:


rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[145]:


rf_random.fit(X_train,y_train)


# In[146]:


rf_random.best_params_


# In[147]:


prediction = rf_random.predict(X_test)


# In[148]:


plt.figure(figsize = (8,8))
sns.distplot(y_test-prediction)
plt.show()


# In[149]:


plt.figure(figsize = (8,8))
plt.scatter(y_test, prediction, alpha = 0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[150]:


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[ ]:




