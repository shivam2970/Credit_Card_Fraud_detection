#!/usr/bin/env python
# coding: utf-8

# # credit card fraud detection using unsupervised machine learning technique

# Importing neccesary libraries 

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


pip install seaborn 


# In[5]:


import seaborn as sns


# importing our datasets and naming it as "data"

# In[10]:


data=pd.read_csv("creditcard.csv")
data.head()


# In[11]:


data.isnull().sum()


# In[12]:


### from above we can see that there ois zero null value to we can move on further with our datasets


# In[13]:


data.shape


# In[14]:


data.describe()


# In[15]:


a=np.arange(1,284808)


# In[18]:


ax=plt.figure(figsize=(12,6))
sns.barplot(x='Class',y=a,data=data,palette='Paired')
plt.title('Frequency distribution of class')
plt.ylabel('Frequency')
plt.xlabel('Class')


# In[22]:


data.corr()


# In[28]:


fraud=data['Class'].value_counts()
fraud_percentage= (fraud/284807)*100
fraud_percentage


# In[29]:


### from above we can say that there are almost 0.173% fraud transaction i.e 0.173*284807=49271


# In[33]:


x=data["Amount"]
y=data["Class"]


# In[ ]:





# In[37]:


X=x.values.reshape(-1,1)


# In[38]:


x.shape


# In[39]:


fig,ax=plt.subplots(figsize=(20,6))
ax.scatter(X,data['Time'])
plt.title('Amount taken with seconds elapsed between each transaction and the first transaction in the dataset ')
plt.xlabel('Amount')
plt.ylabel('Time')
plt.show()


# In[40]:


from sklearn.preprocessing  import StandardScaler
sdc= StandardScaler()


# In[41]:


data_scaled = pd.DataFrame(sdc.fit_transform(data), columns=data.columns)


# In[42]:


from sklearn.model_selection import train_test_split


# In[44]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[45]:


### as this is unbaalnced datasets we are using insupervised learning algorthim i.e logistic regression


# In[46]:


from sklearn.linear_model import LogisticRegression


# In[47]:


lr=LogisticRegression()


# In[48]:


lr.fit(X_train, y_train)


# In[49]:


pred=lr.predict(X_test)
pred


# In[50]:


### we will use confusion matrix for checking performance of our classification model


# In[51]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm


# In[52]:


from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,pred)
score*100


# In[53]:


#### so our model is 99.84% accurate


# In[ ]:




