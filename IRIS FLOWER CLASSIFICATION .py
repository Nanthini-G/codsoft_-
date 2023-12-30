#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') 


# In[2]:


df = pd.read_csv('IRIS.csv')
df.head() 


# In[4]:


df.describe() 


# In[5]:


df.info() 


# In[7]:


df['species'].value_counts()  


# In[8]:


df.isnull().sum() 


# In[10]:


df['sepal_length'].hist() 


# In[11]:


df['sepal_width'].hist() 


# In[12]:


df['petal_length'].hist() 


# In[14]:


colors = ['m', 'c', 'b'] 
species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']


# In[17]:


for i in range(3):
    # filter data on each class
    x = df[df['species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['sepal_length'], x['sepal_width'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend() 


# In[19]:


for i in range(3):
    # filter data on each class
    x = df[df['species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['petal_length'], x['petal_width'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend() 


# In[20]:


for i in range(3):
    # filter data on each class
    x = df[df['species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['sepal_length'], x['petal_length'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend() 


# In[21]:


for i in range(3):
    # filter data on each class
    x = df[df['species'] == species[i]]
    # plot the scatter plot
    plt.scatter(x['sepal_width'], x['petal_width'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend() 


# In[22]:


df.corr() 


# In[29]:


corr = df.corr()
# plot the heat map
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax, cmap = 'coolwarm')   


# In[30]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# transform the string labels to integer
df['species'] = le.fit_transform(df['species'])
df.head() 


# In[34]:


from sklearn.model_selection import train_test_split
X = df.drop(columns=['species'])
Y = df['species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# In[35]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression() 


# In[36]:


model.fit(x_train, y_train) 


# In[37]:


print("Accuracy: ",model.score(x_test, y_test) * 100) 


# In[ ]:




