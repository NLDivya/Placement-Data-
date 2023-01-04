#!/usr/bin/env python
# coding: utf-8

# In[14]:


#importing libraries
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression


# In[2]:


#load the data
df = pd.read_csv('placement.csv')


# In[7]:


#understand the data
df.head(22)


# In[4]:


df.info()


# #### No null values found. The dataset is clean

# In[5]:


df.shape


# In[8]:


df.describe()


# In[11]:


df['placed'].value_counts()


# #### Out of 1000 students, 489 are placed 511 are not placed

# In[ ]:


#Visualize the data


# In[24]:


#outliers 
sb.boxplot(df['cgpa'], color = 'purple')


# ### outliers are present in cgpa at 4.5, 5.5 and above 8.5

# In[21]:


sb.boxplot(df['placement_exam_marks'], color = 'green')
plt.xlabel('Package Received')


# #### observation: people have mostly have been offered 28 LPA 
# ####                  people have also been offered more than 80 LPA to 100 LPA

# In[26]:


df.corr()


# In[27]:


#corelation
#the cgpa and package received are negatively/weakly corelated


# In[43]:


plt.figure(figsize=(7,8))
plt.xlabel('CGPA')
plt.ylabel('Package Offered')
sb.scatterplot( data = df, x = 'cgpa', y='placement_exam_marks', hue = 'placed')


# In[69]:


#heatmap
sb.heatmap(df.corr(), annot = True)


# In[44]:


# data points are scattered randomly
# regression problem target variable is placement_exam_marks which is a numerical variable


# In[45]:


X = df['cgpa']


# In[46]:


y = df['placement_exam_marks']


# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.70, test_size = 0.30, random_state = 42)


# In[50]:


model = LinearRegression()


# In[56]:


X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)


# In[57]:


model.fit(X_train, y_train)


# In[58]:


y_pred = model.predict(X_test)


# In[72]:


from sklearn.metrics import mean_squared_error, r2_score


# In[60]:


mean_squared_error(y_pred, y_test)


# In[61]:


from sklearn.tree import DecisionTreeRegressor


# In[62]:


model_1 = DecisionTreeRegressor(max_leaf_nodes=5)


# In[63]:


model_1.fit(X_train, y_train)


# In[65]:


y_pred_1 = model_1.predict(X_test)


# In[66]:


mean_squared_error(y_pred_1, y_test)


# In[73]:


print("Mean squared error(MSE): %.2f" % mean_squared_error(y_pred_1, y_test))
    # Checking the R2 value
print("Coefficient of determination: %.3f" % r2_score(y_pred_1, y_test))


# In[ ]:




