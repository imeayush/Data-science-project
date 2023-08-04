#!/usr/bin/env python
# coding: utf-8

# # 8- Uplift Modeling

# In[1]:


from datetime import datetime, timedelta,date
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from __future__ import division
from sklearn.cluster import KMeans


# In[2]:


import plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go


# In[3]:


import sklearn
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import warnings
warnings.filterwarnings("ignore")


# In[4]:


pyoff.init_notebook_mode()


# In[5]:


def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


# In[6]:


df_data = pd.read_csv('data.csv')


# In[7]:


df_data.head(10)


# In[8]:


df_data.info()


# In[9]:


df_data.conversion.mean()


# In[10]:


def calc_uplift(df):
    avg_order_value = 25
    
    #calculate conversions for each offer type
    base_conv = df[df.offer == 'No Offer']['conversion'].mean()
    disc_conv = df[df.offer == 'Discount']['conversion'].mean()
    bogo_conv = df[df.offer == 'Buy One Get One']['conversion'].mean()
    
    #calculate conversion uplift for discount and bogo
    disc_conv_uplift = disc_conv - base_conv
    bogo_conv_uplift = bogo_conv - base_conv
    
    #calculate order uplift
    disc_order_uplift = disc_conv_uplift * len(df[df.offer == 'Discount']['conversion'])
    bogo_order_uplift = bogo_conv_uplift * len(df[df.offer == 'Buy One Get One']['conversion'])
    
    #calculate revenue uplift
    disc_rev_uplift = disc_order_uplift * avg_order_value
    bogo_rev_uplift = bogo_order_uplift * avg_order_value
    
    
    print('Discount Conversion Uplift: {0}%'.format(np.round(disc_conv_uplift*100,2)))
    print('Discount Order Uplift: {0}'.format(np.round(disc_order_uplift,2)))
    print('Discount Revenue Uplift: ${0}\n'.format(np.round(disc_rev_uplift,2)))
    
    if len(df[df.offer == 'Buy One Get One']['conversion']) > 0:
          
        print('-------------- \n')
        print('BOGO Conversion Uplift: {0}%'.format(np.round(bogo_conv_uplift*100,2)))
        print('BOGO Order Uplift: {0}'.format(np.round(bogo_order_uplift,2)))
        print('BOGO Revenue Uplift: ${0}'.format(np.round(bogo_rev_uplift,2)))     


# In[11]:


calc_uplift(df_data)


# In[12]:


df_data['campaign_group'] = 'treatment'
df_data.loc[df_data.offer == 'No Offer', 'campaign_group'] = 'control'


# In[13]:


df_data['target_class'] = 0 #CN
df_data.loc[(df_data.campaign_group == 'control') & (df_data.conversion > 0),'target_class'] = 1 #CR
df_data.loc[(df_data.campaign_group == 'treatment') & (df_data.conversion == 0),'target_class'] = 2 #TN
df_data.loc[(df_data.campaign_group == 'treatment') & (df_data.conversion > 0),'target_class'] = 3 #TR


# In[14]:


df_data.target_class.value_counts()


# In[15]:


df_data.target_class.value_counts()/len(df_data)


# In[16]:


#creating the clusters
kmeans = KMeans(n_clusters=5)
kmeans.fit(df_data[['history']])
df_data['history_cluster'] = kmeans.predict(df_data[['history']])


# In[17]:


df_data = order_cluster('history_cluster', 'history',df_data,True)


# In[18]:


df_model = df_data.drop(['offer','campaign_group','conversion'],axis=1)
df_model = pd.get_dummies(df_model)


# In[19]:


df_model.head()


# In[20]:


#create feature set and labels
X = df_model.drop(['target_class'],axis=1)
y = df_model.target_class


# In[21]:


X.columns


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)


# In[24]:


xgb_model = xgb.XGBClassifier().fit(X_train, y_train)


# In[25]:


class_probs = xgb_model.predict_proba(X_test)


# In[26]:


class_probs[0]


# In[27]:


X_test['proba_CN'] = class_probs[:,0] 
X_test['proba_CR'] = class_probs[:,1] 
X_test['proba_TN'] = class_probs[:,2] 
X_test['proba_TR'] = class_probs[:,3] 


# In[28]:


X_test['uplift_score'] = X_test.eval('proba_CN + proba_TR - proba_TN - proba_CR')


# In[29]:


X_test.head()


# In[30]:


overall_proba = xgb_model.predict_proba(df_model.drop(['target_class'],axis=1))


# In[31]:


df_model['proba_CN'] = overall_proba[:,0] 
df_model['proba_CR'] = overall_proba[:,1] 
df_model['proba_TN'] = overall_proba[:,2] 
df_model['proba_TR'] = overall_proba[:,3] 


# In[32]:


df_model['uplift_score'] = df_model.eval('proba_CN + proba_TR - proba_TN - proba_CR')


# In[33]:


df_data['uplift_score'] = df_model['uplift_score']


# In[34]:


df_data.head()


# In[35]:


df_data.groupby('offer').uplift_score.mean()


# In[36]:


df_data_lift = df_data.copy()
uplift_q_75 = df_data_lift.uplift_score.quantile(0.75)
df_data_lift = df_data_lift[(df_data_lift.offer != 'Buy One Get One') & (df_data_lift.uplift_score > uplift_q_75)].reset_index(drop=True)


# In[37]:


len(df_data_lift)


# In[38]:


calc_uplift(df_data_lift)


# In[39]:


df_data_lift = df_data.copy()
uplift_q_5 = df_data_lift.uplift_score.quantile(0.5)
df_data_lift = df_data_lift[(df_data_lift.offer != 'Buy One Get One') & (df_data_lift.uplift_score < uplift_q_5)].reset_index(drop=True)


# In[40]:


calc_uplift(df_data_lift)


# In[ ]:




