
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy import stats
import seaborn as sns
recpies = pd.read_csv("Z:\data analytics\epi_r.csv")


# In[2]:


recpies.dropna(inplace = True)


# In[3]:


recpies = recpies[recpies["calories"] <= 10000]
print("Is this variable numeric?")
np.issubdtype(recpies['rating'].dtype, np.number)


# In[4]:


print("Is this variable an integer?")
np.issubdtype(recpies['rating'].dtype, np.integer)


# In[5]:


plt.figure(figsize=(10,10))
plt.scatter(recpies['calories'], recpies['dessert'])


# In[6]:


X = recpies[["calories"]]
y = recpies["dessert"]
clf = linear_model.LogisticRegression()
clf.fit(X, y)


# In[7]:


X_test = np.linspace(0, 10000)


# In[8]:


def model(x):
    return 1 / (1 + np.exp(-x))


# In[9]:


loss = model(X_test * clf.coef_ + clf.intercept_).ravel()
plt.figure(figsize=(10,10))
plt.scatter(recpies['calories'], recpies['dessert'])
plt.plot(X_test, loss, color='red', linewidth=3)


# In[10]:


sns.lmplot(x="calories", y="dessert", data=recpies,
           logistic=True, size=10);

