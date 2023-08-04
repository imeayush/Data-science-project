#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().system('pip install xgboost')


# In[12]:


encoding = 'latin-1'


# In[13]:


filename = r'C:\Users\Ayush Lokhande\OnlineRetail.csv'


# In[14]:


from datetime import datetime, timedelta
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


import plotly.offline as pyoff
import plotly.graph_objs as go


# In[16]:


from __future__ import division
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split


# In[17]:


import xgboost as xgb


# In[18]:


pyoff.init_notebook_mode()


# In[19]:


tx_data = pd.read_csv('OnlineRetail.csv',encoding = "ISO-8859-1")
tx_data.head(10)


# In[20]:


tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])


# In[21]:


tx_data['InvoiceDate'].describe()


# In[22]:


tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)


# In[27]:


date(2007, 12, 5)


# In[29]:


tx_3m = tx_uk[(tx_uk.InvoiceDate < datetime(2011, 6, 1)) & (tx_uk.InvoiceDate >= datetime(2011, 3, 1))].reset_index(drop=True)
tx_6m = tx_uk[(tx_uk.InvoiceDate >= datetime(2011, 6, 1)) & (tx_uk.InvoiceDate < datetime(2011, 12, 1))].reset_index(drop=True)


# In[30]:


tx_3m['InvoiceDate'].describe()


# In[31]:


tx_user = pd.DataFrame(tx_3m['CustomerID'].unique())
tx_user.columns = ['CustomerID']


# # Recency

# In[32]:


tx_max_purchase = tx_3m.groupby('CustomerID').InvoiceDate.max().reset_index()


# In[33]:


tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']


# In[34]:


tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days


# In[35]:


tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')


# In[41]:


tx_user.head()


# In[42]:


tx_user.Recency.describe()


# In[43]:


plot_data = [
    go.Histogram(
        x=tx_user['Recency']
    )
]

plot_layout = go.Layout(
        title='Recency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[44]:


sse={}
tx_recency = tx_user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_recency)
    tx_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()


# In[46]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])


# In[47]:


tx_user.groupby('RecencyCluster')['Recency'].describe()


# In[48]:


def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


# In[49]:


tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)


# # Frequency

# In[50]:


tx_frequency = tx_3m.groupby('CustomerID').InvoiceDate.count().reset_index()


# In[51]:


tx_frequency.columns = ['CustomerID','Frequency']


# In[52]:


tx_frequency.head()


# In[53]:


tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')


# In[54]:


tx_user.head()


# In[55]:


tx_user.Frequency.describe()


# In[56]:


plot_data = [
    go.Histogram(
        x=tx_user.query('Frequency < 1000')['Frequency']
    )
]

plot_layout = go.Layout(
        title='Frequency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[57]:


sse={}
tx_frequency = tx_user[['Frequency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_frequency)
    tx_frequency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()


# In[58]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Frequency']])
tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])


# In[59]:


tx_user.groupby('FrequencyCluster')['Frequency'].describe()


# In[60]:


tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)


# # Monetary Value

# In[61]:


tx_3m['Revenue'] = tx_3m['UnitPrice'] * tx_3m['Quantity']


# In[62]:


tx_revenue = tx_3m.groupby('CustomerID').Revenue.sum().reset_index()


# In[63]:


tx_revenue.head()


# In[64]:


tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')


# In[65]:


tx_user.Revenue.describe()


# In[66]:


plot_data = [
    go.Histogram(
        x=tx_user.query('Revenue < 10000')['Revenue']
    )
]

plot_layout = go.Layout(
        title='Monetary Value'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[67]:


import warnings
warnings.filterwarnings("ignore")


# In[68]:


sse={}
tx_revenue = tx_user[['Revenue']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_revenue)
    tx_revenue["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()


# In[69]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Revenue']])
tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])


# In[70]:


tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)


# In[71]:


tx_user.groupby('RevenueCluster')['Revenue'].describe()


# # Overall Segmentation

# In[72]:


tx_user.head()


# In[73]:


tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']


# In[74]:


tx_user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()


# In[77]:


tx_user.groupby('OverallScore')['Recency'].count()


# In[78]:


tx_user['Segment'] = 'Low-Value'
tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value' 


# In[83]:


tx_graph = tx_user.query("Revenue < 50000 and Frequency < 2000")

plot_data = [
    go.Scatter(
        x=tx_graph.query("Segment == 'Low-Value'")['Frequency'],
        y=tx_graph.query("Segment == 'Low-Value'")['Revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'green',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Mid-Value'")['Frequency'],
        y=tx_graph.query("Segment == 'Mid-Value'")['Revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'High-Value'")['Frequency'],
        y=tx_graph.query("Segment == 'High-Value'")['Revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'pink',
            opacity= 0.9
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "Revenue"},
        xaxis= {'title': "Frequency"},
        title='Segments'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[85]:


tx_graph = tx_user.query("Revenue < 50000 and Frequency < 2000")

plot_data = [
    go.Scatter(
        x=tx_graph.query("Segment == 'Low-Value'")['Frequency'],
        y=tx_graph.query("Segment == 'Low-Value'")['Revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Mid-Value'")['Frequency'],
        y=tx_graph.query("Segment == 'Mid-Value'")['Revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'High-Value'")['Frequency'],
        y=tx_graph.query("Segment == 'High-Value'")['Revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "Revenue"},
        xaxis= {'title': "Frequency"},
        title='Segments'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[86]:


tx_graph = tx_user.query("Revenue < 50000 and Frequency < 2000")

plot_data = [
    go.Scatter(
        x=tx_graph.query("Segment == 'Low-Value'")['Recency'],
        y=tx_graph.query("Segment == 'Low-Value'")['Frequency'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Mid-Value'")['Recency'],
        y=tx_graph.query("Segment == 'Mid-Value'")['Frequency'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'High-Value'")['Recency'],
        y=tx_graph.query("Segment == 'High-Value'")['Frequency'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "Frequency"},
        xaxis= {'title': "Recency"},
        title='Segments'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# # LTV

# In[87]:


tx_user.head()


# In[88]:


tx_6m.head()


# In[89]:


tx_6m['Revenue'] = tx_6m['UnitPrice'] * tx_6m['Quantity']


# In[90]:


tx_user_6m = tx_6m.groupby('CustomerID')['Revenue'].sum().reset_index()
tx_user_6m.columns = ['CustomerID','m6_Revenue']


# In[91]:


tx_6m.head()


# In[92]:


plot_data = [
    go.Histogram(
        x=tx_user_6m.query('m6_Revenue < 10000')['m6_Revenue']
    )
]

plot_layout = go.Layout(
        title='6m Revenue'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[93]:


tx_merge = pd.merge(tx_user, tx_user_6m, on='CustomerID', how='left')


# In[94]:


tx_merge.head()


# In[95]:


tx_merge = tx_merge.fillna(0)


# In[96]:


tx_merge.groupby('Segment')['m6_Revenue'].mean()


# In[97]:


tx_graph = tx_merge.query("m6_Revenue < 30000")

plot_data = [
    go.Scatter(
        x=tx_graph.query("Segment == 'Low-Value'")['OverallScore'],
        y=tx_graph.query("Segment == 'Low-Value'")['m6_Revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Mid-Value'")['OverallScore'],
        y=tx_graph.query("Segment == 'Mid-Value'")['m6_Revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'High-Value'")['OverallScore'],
        y=tx_graph.query("Segment == 'High-Value'")['m6_Revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "6m LTV"},
        xaxis= {'title': "RFM Score"},
        title='LTV'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[98]:


tx_merge = tx_merge[tx_merge['m6_Revenue']<tx_merge['m6_Revenue'].quantile(0.99)]


# In[99]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(tx_merge[['m6_Revenue']])
tx_merge['LTVCluster'] = kmeans.predict(tx_merge[['m6_Revenue']])


# In[100]:


tx_merge = order_cluster('LTVCluster', 'm6_Revenue',tx_merge,True)


# In[102]:


tx_merge.groupby('LTVCluster')['m6_Revenue'].describe()


# In[103]:


tx_cluster = tx_merge.copy()


# In[105]:


tx_cluster.head()


# In[109]:


tx_cluster.groupby('LTVCluster')['m6_Revenue'].describe()


# In[110]:


tx_cluster.head()


# In[112]:


tx_class = pd.get_dummies(tx_cluster)


# In[113]:


tx_class.head()


# In[114]:


corr_matrix = tx_class.corr()
corr_matrix['LTVCluster'].sort_values(ascending=False)


# In[115]:


X = tx_class.drop(['LTVCluster','m6_Revenue'],axis=1)
y = tx_class['LTVCluster']


# In[116]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)


# In[117]:


ltv_xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1,objective= 'multi:softprob',n_jobs=-1).fit(X_train, y_train)

print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(ltv_xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(ltv_xgb_model.score(X_test[X_train.columns], y_test)))


# In[118]:


tx_class.groupby('LTVCluster').CustomerID.count()/tx_class.CustomerID.count()


# In[119]:


y_pred = ltv_xgb_model.predict(X_test)


# In[123]:


from sklearn.metrics import classification_report


# In[125]:


print(classification_report(y_test, y_pred))


# In[ ]:




