#!/usr/bin/env python
# coding: utf-8

# # Linear Modeling Assignment

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('regrex1.csv')


# In[3]:


df.head()


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


df['x'].head()


# In[6]:


df['y'].head()


# In[19]:


plt.scatter(df['x'], df['y'])
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Scatter')
plt.show


# In[20]:


get_ipython().system('pip install scikit-learn')


# In[21]:


import numpy as np
from sklearn.linear_model import LinearRegression


# In[22]:


x = np.array(df['x']).reshape((-1,1))
y = np.array(df['y'])


# In[23]:


model = LinearRegression()


# In[24]:


model.fit(x,y)


# In[25]:


intercept = model.intercept_
slope = model.coef_
r_sq = model.score(x,y)


# In[26]:


print(f"intercept: {intercept}")
print(f"slope: {slope}")
print(f"r squared: {r_sq}")


# In[27]:


y_pred = model.predict(x)


# In[28]:


y_pred


# In[29]:


plt.plot(df['x'], y_pred)
plt.show()


# In[30]:


plt.scatter(df['x'], df['y'])
plt.plot(df['x'], y_pred)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Linear Regression')
plt.show()


# In[ ]:





# In[ ]:




