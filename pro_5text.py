#!/usr/bin/env python
# coding: utf-8

# In[1]:


encoding = 'latin-1'


# In[2]:


filename = r'C:\Users\Ayush Lokhande\OnlineRetail.csv'


# In[4]:


from datetime import datetime, timedelta,date
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from __future__ import division
from sklearn.cluster import KMeans


# In[5]:


import warnings
warnings.filterwarnings("ignore")


# In[7]:


import plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go


# In[8]:


from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split


# In[9]:


import xgboost as xgb


# In[10]:


pyoff.init_notebook_mode()


# In[11]:


tx_data = pd.read_csv('OnlineRetail.csv',encoding = "ISO-8859-1")
tx_data.head(10)


# In[21]:


tx_uk['InvoiceDate'] = pd.to_datetime(tx_uk['InvoiceDate']).dt.date


# In[22]:


tx_6m = tx_uk[(tx_uk.InvoiceDate < date(2011, 9, 1)) & (tx_uk.InvoiceDate >= date(2011, 3, 1))].reset_index(drop=True)
tx_next = tx_uk[(tx_uk.InvoiceDate >= date(2011, 9, 1)) & (tx_uk.InvoiceDate < date(2011, 12, 1))].reset_index(drop=True)


# In[23]:


tx_next['InvoiceDate'].describe()


# In[24]:


tx_user = pd.DataFrame(tx_6m['CustomerID'].unique())
tx_user.columns = ['CustomerID']


# # Adding label¶

# In[25]:


tx_next_first_purchase = tx_next.groupby('CustomerID').InvoiceDate.min().reset_index()


# In[26]:


tx_next_first_purchase.columns = ['CustomerID','MinPurchaseDate']


# In[27]:


tx_next_first_purchase.head()


# In[28]:


tx_last_purchase = tx_6m.groupby('CustomerID').InvoiceDate.max().reset_index()


# In[29]:


tx_last_purchase.columns = ['CustomerID','MaxPurchaseDate']


# In[30]:


tx_purchase_dates = pd.merge(tx_last_purchase,tx_next_first_purchase,on='CustomerID',how='left')


# In[31]:


tx_purchase_dates['NextPurchaseDay'] = (tx_purchase_dates['MinPurchaseDate'] - tx_purchase_dates['MaxPurchaseDate']).dt.days


# In[32]:


tx_purchase_dates.head()


# In[43]:


tx_user.NextPurchaseDay.describe()


# In[33]:


tx_user = pd.merge(tx_user, tx_purchase_dates[['CustomerID','NextPurchaseDay']],on='CustomerID',how='left')


# In[34]:


tx_user.head()


# In[35]:


tx_user.shape


# In[36]:


tx_user = tx_user.fillna(999)


# # Recency

# In[37]:


tx_max_purchase = tx_6m.groupby('CustomerID').InvoiceDate.max().reset_index()


# In[38]:


tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']


# In[39]:


tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days


# In[40]:


tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')


# In[41]:


tx_user.head()


# In[42]:


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


# In[45]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])


# In[46]:


def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


# In[47]:


tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)


# In[48]:


tx_user.groupby('RecencyCluster')['Recency'].describe()


# # Frequency
# 

# In[49]:


tx_frequency = tx_6m.groupby('CustomerID').InvoiceDate.count().reset_index()


# In[50]:


tx_frequency.columns = ['CustomerID','Frequency']


# In[51]:


tx_frequency.head()


# In[52]:


tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')


# In[53]:


tx_user.head()


# In[54]:


tx_user.Frequency.describe()


# In[57]:


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


# In[58]:


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


# In[59]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Frequency']])
tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])


# In[60]:


tx_user.groupby('FrequencyCluster')['Frequency'].describe()


# In[61]:


tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)


# # Monetary Value

# In[62]:


# calculate monetary value, create a dataframe with it


# In[63]:


tx_6m['Revenue'] = tx_6m['UnitPrice'] * tx_6m['Quantity']
tx_revenue = tx_6m.groupby('CustomerID').Revenue.sum().reset_index()


# In[64]:


# now we see the dataframe as a describe form 


# In[65]:


tx_revenue.head()


# In[66]:


# add Revenue column to tx_user


# In[67]:


tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')


# In[68]:


tx_user.Revenue.describe()


# In[69]:


# now plot revenue 


# In[70]:


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


# In[71]:


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


# In[72]:


#revenue cluster 


# In[73]:


kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Revenue']])
tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])


# In[75]:


#ordering clusters and who the characteristics


# In[76]:


tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)
tx_user.groupby('RevenueCluster')['Revenue'].describe()


# # building overall segmentation

# In[77]:


tx_user.head()


# In[78]:


tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']


# In[79]:


tx_user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()


# In[80]:


tx_user.groupby('OverallScore')['Recency'].count()


# In[81]:


tx_user['Segment'] = 'Low-Value'
tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value' 


# In[ ]:


# now we would see thefrecuency ,  resenancy and more frecuency


# In[82]:


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


# In[84]:


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


# In[85]:


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


# In[86]:


tx_user.head()


# # Adding new features

# In[87]:


tx_6m.head()


# In[88]:


#create a dataframe with CustomerID and Invoice Date


# In[89]:


tx_day_order = tx_6m[['CustomerID','InvoiceDate']]


# In[90]:


#Convert Invoice Datetime to day


# In[92]:


tx_day_order['InvoiceDay'] = pd.to_datetime(tx_6m['InvoiceDate']).dt.date


# In[93]:


tx_day_order = tx_day_order.sort_values(['CustomerID','InvoiceDate'])


# In[94]:


#Drop duplicates
tx_day_order = tx_day_order.drop_duplicates(subset=['CustomerID','InvoiceDay'],keep='first')


# In[95]:


#shifting last 3 purchase dates
tx_day_order['PrevInvoiceDate'] = tx_day_order.groupby('CustomerID')['InvoiceDay'].shift(1)
tx_day_order['T2InvoiceDate'] = tx_day_order.groupby('CustomerID')['InvoiceDay'].shift(2)
tx_day_order['T3InvoiceDate'] = tx_day_order.groupby('CustomerID')['InvoiceDay'].shift(3)


# In[96]:


tx_day_order.head()


# # calculating the difference in days for each invoice date

# In[97]:


tx_day_order['DayDiff'] = (tx_day_order['InvoiceDay'] - tx_day_order['PrevInvoiceDate']).dt.days
tx_day_order['DayDiff2'] = (tx_day_order['InvoiceDay'] - tx_day_order['T2InvoiceDate']).dt.days
tx_day_order['DayDiff3'] = (tx_day_order['InvoiceDay'] - tx_day_order['T3InvoiceDate']).dt.days


# In[98]:


tx_day_order.head(10)


# # we utilize .agg() method to find out the mean

# In[99]:


tx_day_diff = tx_day_order.groupby('CustomerID').agg({'DayDiff': ['mean','std']}).reset_index()


# In[100]:


tx_day_diff.columns = ['CustomerID', 'DayDiffMean','DayDiffStd']


# In[103]:


tx_day_diff.head()


# In[101]:


# now we find >3 purchases of custoumer 


# In[102]:


tx_day_order_last = tx_day_order.drop_duplicates(subset=['CustomerID'],keep='last')


# In[105]:


tx_day_order_last.head(10)


# In[106]:


tx_day_diff = tx_day_order.groupby('CustomerID').agg({'DayDiff': ['mean','std']}).reset_index()


# In[107]:


tx_day_diff.columns = ['CustomerID', 'DayDiffMean','DayDiffStd']


# In[108]:


tx_day_diff.head()


# In[109]:


tx_day_order_last = tx_day_order.drop_duplicates(subset=['CustomerID'],keep='last')


# In[110]:


tx_day_order_last.head(10)


# In[111]:


tx_day_order_last = tx_day_order_last.dropna()


# In[112]:


tx_day_order_last = pd.merge(tx_day_order_last, tx_day_diff, on='CustomerID')


# In[113]:


tx_user = pd.merge(tx_user, tx_day_order_last[['CustomerID','DayDiff','DayDiff2','DayDiff3','DayDiffMean','DayDiffStd']], on='CustomerID')


# In[114]:


tx_user.head()


# In[116]:


len(tx_user)


# # Grouping the label

# In[117]:


tx_class = tx_user.copy()


# In[118]:


tx_class = pd.get_dummies(tx_class)


# In[119]:


tx_class.tail(10)


# # we need to identify the classes in our label

# In[120]:


tx_user.NextPurchaseDay.describe()


# In[122]:


tx_class['NextPurchaseDayRange'] = 2
tx_class.loc[tx_class.NextPurchaseDay>20,'NextPurchaseDayRange'] = 1
tx_class.loc[tx_class.NextPurchaseDay>50,'NextPurchaseDayRange'] = 0


# In[123]:


tx_class.NextPurchaseDayRange.value_counts()/len(tx_user)


# In[124]:


#tx_class = tx_class[tx_class.Frequency>10]


# In[127]:


#The last step is to see the correlation between our features and label


#  #   The correlation matrix

# In[126]:


corr = tx_class[tx_class.columns].corr()
plt.figure(figsize = (30,30))
sns.heatmap(corr, annot = True, linewidths=0.2, fmt=".2f")


# In[128]:


# train and test split 


# In[129]:


tx_class = tx_class.drop('NextPurchaseDay',axis=1)


# In[130]:


len(tx_class)


# In[131]:


X, y = tx_class.drop('NextPurchaseDayRange',axis=1), tx_class.NextPurchaseDayRange
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)


# In[132]:


# create an array of models 


# In[133]:


models = []
models.append(("LR",LogisticRegression()))
models.append(("NB",GaussianNB()))
models.append(("RF",RandomForestClassifier()))
models.append(("SVC",SVC()))
models.append(("Dtree",DecisionTreeClassifier()))
models.append(("XGB",xgb.XGBClassifier()))
models.append(("KNN",KNeighborsClassifier()))


# In[134]:


# measure the accuracy 


# In[160]:


from sklearn.model_selection import StratifiedKFold, cross_val_score

for name, model in models:
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=22)
    cv_result = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
    print(name, cv_result)


# In[165]:


from sklearn.model_selection import KFold, cross_val_score


# In[169]:


#######we face some error on it #########
for name,model in models:
    kfold = KFold(n_splits=2, random_state=22)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
    print(name, cv_result)


# # Multi-Classification Model

# In[170]:


xgb_model = xgb.XGBClassifier().fit(X_train, y_train)

print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(xgb_model.score(X_test[X_train.columns], y_test)))


# In[174]:


y_pred = xgb_model.predict(X_test)


# In[175]:


from sklearn.metrics import classification_report


# In[176]:


print(classification_report(y_test, y_pred))


# In[179]:


from sklearn.model_selection import GridSearchCV
import xgboost as xgb

param_test1 = {
 'max_depth': range(3, 10, 2),
 'min_child_weight': range(1, 6, 2)
}

gsearch1 = GridSearchCV(estimator=xgb.XGBClassifier(), param_grid=param_test1, scoring='accuracy', n_jobs=-1, cv=2)
gsearch1.fit(X_train, y_train)
print(gsearch1.best_params_, gsearch1.best_score_)


# In[180]:


xgb_model = xgb.XGBClassifier(max_depth=3, min_child_weight=5).fit(X_train, y_train)

print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(xgb_model.score(X_test[X_train.columns], y_test)))


# In[181]:


y_pred = xgb_model.predict(X_test)


# In[182]:


print classification_report(y_test, y_pred)


# In[ ]:




