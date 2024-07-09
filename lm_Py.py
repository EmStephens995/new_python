#!/usr/bin/env python
# coding: utf-8

# # Linear Modeling Assignment

# In[6]:


import pandas as pd


# In[7]:


df = pd.read_csv('regrex1.csv')


# In[8]:


df.head()


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


df['x'].head()


# In[11]:


df['y'].head()


# In[12]:


plt.scatter(df['x'], df['y'])
plt.show


# In[13]:


get_ipython().system('pip install scikit-learn')


# In[14]:


import numpy as np
from sklearn.linear_model import LinearRegression


# In[15]:


x = np.array(df['x']).reshape((-1,1))
y = np.array(df['y'])


# In[16]:


model = LinearRegression()


# In[17]:


model.fit(x,y)


# In[18]:


intercept = model.intercept_
slope = model.coef_
r_sq = model.score(x,y)


# In[19]:


print(f"intercept: {intercept}")
print(f"slope: {slope}")
print(f"r squared: {r_sq}")


# In[20]:


y_pred = model.predict(x)


# In[21]:


y_pred


# In[22]:


plt.plot(df['x'], y_pred)
plt.show()


# In[28]:


plt.scatter(df['x'], df['y'])
plt.plot(df['x'], y_pred)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Linear Regression')
plt.show()


# In[ ]:





# In[ ]:




