#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings(action='ignore') 


# In[9]:


pd.set_option('display.max_columns',10,'display.width',1000)
test = pd.read_csv('tested.csv')
test.head() 


# In[8]:


test.shape


# In[10]:


test.isnull().sum()


# In[11]:


test.describe(include="all")


# In[12]:


test.groupby('Survived').mean()


# In[13]:


test.corr()


# In[14]:


male_ind = len(test[test['Sex'] == 'male'])
print("No of Males in Titanic:",male_ind)


# In[15]:


female_ind = len(test[test['Sex'] == 'female'])
print("No of Females in Titanic:",female_ind)


# In[17]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
gender = ['Male','Female']
index = [266,152]
ax.bar(gender,index)
plt.xlabel("Gender")
plt.ylabel("No of people onboarding ship")
plt.show()


# In[19]:


alive = len(test[test['Survived'] == 1])
dead = len(test[test['Survived'] == 0])


# In[20]:


test.groupby('Sex')[['Survived']].mean()


# In[21]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
status = ['Survived','Dead']
ind = [alive,dead]
ax.bar(status,ind)
plt.xlabel("Status")
plt.show()


# In[23]:


plt.figure(1)
test.loc[test['Survived'] == 1, 'Pclass'].value_counts().sort_index().plot.bar()
plt.title('Bar graph of people accrding to ticket class in which people survived')


plt.figure(2)
test.loc[test['Survived'] == 0, 'Pclass'].value_counts().sort_index().plot.bar()
plt.title('Bar graph of people accrding to ticket class in which people couldn\'t survive')


# In[24]:


plt.figure(1)
age  = test.loc[test.Survived == 1, 'Age']
plt.title('The histogram of the age groups of the people that had survived')
plt.hist(age, np.arange(0,100,10))
plt.xticks(np.arange(0,100,10))


plt.figure(2)
age  = test.loc[test.Survived == 0, 'Age']
plt.title('The histogram of the age groups of the people that coudn\'t survive')
plt.hist(age, np.arange(0,100,10))
plt.xticks(np.arange(0,100,10)) 


# In[25]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['C = Cherbourg', 'Q = Queenstown', 'S = Southampton']
s = [0.553571,0.389610,0.336957]
ax.pie(s, labels = l,autopct='%1.2f%%')
plt.show()


# In[ ]:




