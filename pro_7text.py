#!/usr/bin/env python
# coding: utf-8

# In[3]:


from datetime import datetime, timedelta,date
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from __future__ import division
from sklearn.cluster import KMeans


# In[4]:


import plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go


# In[5]:


import sklearn
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split


# In[6]:


pyoff.init_notebook_mode()


# In[7]:


def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


# In[8]:


df_data = pd.read_csv('data.csv')


# In[9]:


df_data.head(10)


# In[10]:


df_data.info()


# In[11]:


df_data.conversion.mean()


# In[12]:


def calc_uplift(df):
    #assigning 25$ to the average order value
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
          
    print('-------------- \n')

    print('BOGO Conversion Uplift: {0}%'.format(np.round(bogo_conv_uplift*100,2)))
    print('BOGO Order Uplift: {0}'.format(np.round(bogo_order_uplift,2)))
    print('BOGO Revenue Uplift: ${0}'.format(np.round(bogo_rev_uplift,2)))     


# In[13]:


calc_uplift(df_data)


# In[18]:


df_plot = df_data.groupby('recency').conversion.mean().reset_index()
plot_data = [
    go.Bar(
        x=df_plot['recency'],
        y=df_plot['conversion'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Recency vs Conversion',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)',
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[19]:


kmeans = KMeans(n_clusters=5, n_init='auto')
kmeans.fit(df_data[['history']])
df_data['history_cluster'] = kmeans.predict(df_data[['history']])


# In[20]:


#order the cluster numbers
df_data = order_cluster('history_cluster', 'history',df_data,True)


# In[21]:


df_data.groupby('history_cluster').agg({'history':['mean','min','max'], 'conversion':['count', 'mean']})


# # plot the conversion by each cluster

# In[22]:


df_plot = df_data.groupby('history_cluster').conversion.mean().reset_index()
plot_data = [
    go.Bar(
        x=df_plot['history_cluster'],
        y=df_plot['conversion'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='History vs Conversion',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)',
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# # used_discount & conversion:-

# In[23]:


df_data.groupby(['used_discount','offer']).agg({'conversion':'mean'})


# # Used Discount & BOGO

# In[24]:


df_data.groupby(['used_bogo','offer']).agg({'conversion':'mean'})


# # used_discount	used_bogo	offer	

# In[25]:


df_data.groupby(['used_discount','used_bogo','offer']).agg({'conversion':'mean'})


# In[26]:


#Customers, who used both of the offers before, have the highest conversion rate.


# In[27]:


#Zip Code


# In[29]:


df_plot = df_data.groupby('zip_code').conversion.mean().reset_index()
plot_data = [
    go.Bar(
        x=df_plot['zip_code'],
        y=df_plot['conversion'],
        marker=dict(
        color=['black', 'white', 'red'])
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Zip Code vs Conversion',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)',
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[30]:


#Referral


# In[31]:


df_plot = df_data.groupby('is_referral').conversion.mean().reset_index()
plot_data = [
    go.Bar(
        x=df_plot['is_referral'],
        y=df_plot['conversion'],
        marker=dict(
        color=['green', 'blue', 'orange'])
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Referrals Conversion',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)',
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[32]:


#Channel


# In[33]:


df_plot = df_data.groupby('channel').conversion.mean().reset_index()
plot_data = [
    go.Bar(
        x=df_plot['channel'],
        y=df_plot['conversion'],
        marker=dict(
        color=['green', 'blue', 'orange'])
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Channel vs Conversion',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)',
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[34]:


#Offer Type


# In[35]:


df_plot = df_data.groupby('offer').conversion.mean().reset_index()
plot_data = [
    go.Bar(
        x=df_plot['offer'],
        y=df_plot['conversion'],
        marker=dict(
        color=['green', 'blue', 'orange'])
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Offer vs Conversion',
        plot_bgcolor  = 'rgb(243,243,243)',
        paper_bgcolor  = 'rgb(243,243,243)',
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[36]:


df_model = df_data.copy()
df_model = pd.get_dummies(df_model)


# In[37]:


df_model.head()


# In[38]:


#time to build our machine learning model


# In[39]:


#create feature set and labels
X = df_model.drop(['conversion'],axis=1)
y = df_model.conversion


# In[41]:


X.columns


# In[42]:


#Creating training and test sets:


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)


# In[44]:


xgb_model = xgb.XGBClassifier().fit(X_train, y_train)


# In[45]:


X_test['proba'] = xgb_model.predict_proba(X_test)[:,1] 


# In[46]:


#our probability column looks like:-


# In[47]:


X_test.head(5)


# In[48]:


X_test.proba.mean()


# In[49]:


y_test.mean()


# In[50]:


X_test['conversion'] = y_test


# In[51]:


X_test[X_test['offer_Buy One Get One'] == 1].conversion.mean()


# In[52]:


X_test[X_test['offer_Buy One Get One'] == 1].proba.mean()


# In[53]:


X_test[X_test['offer_Discount'] == 1].conversion.mean()


# In[54]:


X_test[X_test['offer_Discount'] == 1].proba.mean()


# In[55]:


X_test[X_test['offer_No Offer'] == 1].proba.mean()


# In[56]:


real_disc_uptick = len(X_test)*(X_test[X_test['offer_Discount'] == 1].conversion.mean() - X_test[X_test['offer_No Offer'] == 1].conversion.mean())


# In[57]:


pred_disc_uptick = len(X_test)*(X_test[X_test['offer_Discount'] == 1].proba.mean() - X_test[X_test['offer_No Offer'] == 1].proba.mean())


# In[58]:


print('Real Discount Uptick - Order: {}, Revenue: {}'.format(real_disc_uptick, real_disc_uptick*25))
print('Predicted Discount Uptick - Order: {}, Revenue: {}'.format(pred_disc_uptick, pred_disc_uptick*25))


# In[59]:


real_bogo_uptick = int(len(X_test)*(X_test[X_test['offer_Buy One Get One'] == 1].conversion.mean() - X_test[X_test['offer_No Offer'] == 1].conversion.mean()))

pred_bogo_uptick = int(len(X_test)*(X_test[X_test['offer_Buy One Get One'] == 1].proba.mean() - X_test[X_test['offer_No Offer'] == 1].proba.mean()))


# In[60]:


print('Real Discount Uptick - Order: {}, Revenue: {}'.format(real_bogo_uptick, real_bogo_uptick*25))
print('Predicted Discount Uptick - Order: {}, Revenue: {}'.format(pred_bogo_uptick, pred_bogo_uptick*25))


# In[62]:


#Promising results for BOGO:

# Order uptick - real vs predicted: 563 vs 595

# Revenue uptick — real vs predicted: 14075 vs 14875

# The error rate is around 5.6%. The model can benefit from improving the prediction scores on BOGO offer type.


# In[ ]:




