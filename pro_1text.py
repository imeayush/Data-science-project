#!/usr/bin/env python
# coding: utf-8

# In[2]:


encoding = 'latin-1'


# In[3]:


filename = r'C:\Users\Ayush Lokhande\OnlineRetail.csv'


# In[4]:


from datetime import datetime, timedelta
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


import plotly.offline as pyoff
import plotly.graph_objs as go


# In[5]:


pyoff.init_notebook_mode()


# In[6]:


tx_data = pd.read_csv('OnlineRetail.csv',encoding = "ISO-8859-1")
tx_data.head(5)


# In[6]:


tx_data.shape


# In[11]:


tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])


# In[12]:


tx_data['InvoiceDate'].describe()


# In[13]:


tx_data['InvoiceYearMonth'] = tx_data['InvoiceDate'].map(lambda date: 100*date.year + date.month)


# In[12]:


tx_data.head(10)


# In[8]:


tx_data['Revenue'] = tx_data['UnitPrice'] * tx_data['Quantity']


# In[14]:


tx_data.groupby('InvoiceYearMonth')['Revenue'].sum()


# In[15]:


tx_revenue = tx_data.groupby(['InvoiceYearMonth'])['Revenue'].sum().reset_index()


# In[16]:


tx_revenue


# In[17]:


plot_data = [
    go.Scatter(
        x=tx_revenue['InvoiceYearMonth'],
        y=tx_revenue['Revenue'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Montly Revenue'
    )


# In[18]:


fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[19]:


tx_revenue['MonthlyGrowth'] = tx_revenue['Revenue'].pct_change()


# In[19]:


tx_revenue.head()


# In[20]:


plot_data = [
    go.Scatter(
        x=tx_revenue.query("InvoiceYearMonth < 201112")['InvoiceYearMonth'],
        y=tx_revenue.query("InvoiceYearMonth < 201112")['MonthlyGrowth'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Montly Growth Rate'
    )

fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[22]:


tx_data.groupby('Country')['Revenue'].sum().sort_values(ascending=False).astype(int)


# In[23]:


tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)


# In[24]:


tx_uk.head()


# In[25]:


tx_monthly_active = tx_uk.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()


# In[26]:


tx_monthly_active


# In[27]:


plot_data = [
    go.Bar(
        x=tx_monthly_active['InvoiceYearMonth'],
        y=tx_monthly_active['CustomerID'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Active Customers'
    )


# In[28]:


fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[29]:


tx_monthly_active['CustomerID'].mean()


# In[30]:


tx_monthly_sales = tx_uk.groupby('InvoiceYearMonth')['Quantity'].sum().reset_index()


# In[31]:


tx_monthly_sales


# In[32]:


plot_data = [
    go.Bar(
        x=tx_monthly_sales['InvoiceYearMonth'],
        y=tx_monthly_sales['Quantity'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Total # of Order'
    )


# In[33]:


fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[34]:


tx_monthly_sales['Quantity'].mean()


# In[35]:


tx_monthly_order_avg = tx_uk.groupby('InvoiceYearMonth')['Revenue'].mean().reset_index()


# In[36]:


tx_monthly_order_avg


# In[37]:


plot_data = [
    go.Bar(
        x=tx_monthly_order_avg['InvoiceYearMonth'],
        y=tx_monthly_order_avg['Revenue'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Order Average'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[38]:


tx_monthly_order_avg.Revenue.mean()


# In[39]:


tx_uk.info()


# # New & Existing Users

# In[40]:


tx_min_purchase = tx_uk.groupby('CustomerID').InvoiceDate.min().reset_index()


# In[41]:


tx_min_purchase.columns = ['CustomerID','MinPurchaseDate']


# In[43]:


tx_min_purchase['MinPurchaseYearMonth'] = tx_min_purchase['MinPurchaseDate'].map(lambda date: 100*date.year + date.month)


# In[44]:


tx_min_purchase


# In[45]:


tx_uk = pd.merge(tx_uk, tx_min_purchase, on='CustomerID')


# In[46]:


tx_uk.head()


# In[47]:


tx_uk['UserType'] = 'New'
tx_uk.loc[tx_uk['InvoiceYearMonth']>tx_uk['MinPurchaseYearMonth'],'UserType'] = 'Existing'


# In[48]:


tx_uk.UserType.value_counts()


# In[49]:


tx_uk.head()


# In[52]:


tx_user_type_revenue = tx_uk.groupby(['InvoiceYearMonth','UserType'])['Revenue'].sum().reset_index()


# In[53]:


tx_user_type_revenue.query("InvoiceYearMonth != 201012 and InvoiceYearMonth != 201112")


# In[54]:


tx_user_type_revenue = tx_user_type_revenue.query("InvoiceYearMonth != 201012 and InvoiceYearMonth != 201112")


# In[55]:


plot_data = [
    go.Scatter(
        x=tx_user_type_revenue.query("UserType == 'Existing'")['InvoiceYearMonth'],
        y=tx_user_type_revenue.query("UserType == 'Existing'")['Revenue'],
        name = 'Existing'
    ),
    go.Scatter(
        x=tx_user_type_revenue.query("UserType == 'New'")['InvoiceYearMonth'],
        y=tx_user_type_revenue.query("UserType == 'New'")['Revenue'],
        name = 'New'
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='New vs Existing'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[56]:


tx_user_ratio = tx_uk.query("UserType == 'New'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()/tx_uk.query("UserType == 'Existing'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique() 
tx_user_ratio = tx_user_ratio.reset_index()
tx_user_ratio = tx_user_ratio.dropna()


# In[57]:


tx_uk.query("UserType == 'New'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()


# In[58]:


tx_uk.query("UserType == 'Existing'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()


# In[59]:


plot_data = [
    go.Bar(
        x=tx_user_ratio.query("InvoiceYearMonth>201101 and InvoiceYearMonth<201112")['InvoiceYearMonth'],
        y=tx_user_ratio.query("InvoiceYearMonth>201101 and InvoiceYearMonth<201112")['CustomerID'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='New Customer Ratio'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# # Create Signup Data

# In[60]:


tx_min_purchase.head()


# In[61]:


unq_month_year =  tx_min_purchase.MinPurchaseYearMonth.unique()


# In[62]:


unq_month_year


# In[63]:


def generate_signup_date(year_month):
    signup_date = [el for el in unq_month_year if year_month >= el]
    return np.random.choice(signup_date)


# In[67]:


tx_min_purchase['SignupYearMonth'] = tx_min_purchase.apply(lambda row: generate_signup_date(row['MinPurchaseYearMonth']),axis=1)


# In[68]:


tx_min_purchase['InstallYearMonth'] = tx_min_purchase.apply(lambda row: generate_signup_date(row['SignupYearMonth']),axis=1)


# In[70]:


tx_min_purchase.head()


# In[71]:


channels = ['organic','inorganic','referral']


# In[72]:


tx_min_purchase['AcqChannel'] = tx_min_purchase.apply(lambda x: np.random.choice(channels),axis=1)


# # Activation Rate 

# In[73]:


tx_activation = tx_min_purchase[tx_min_purchase['MinPurchaseYearMonth'] == tx_min_purchase['SignupYearMonth']].groupby('SignupYearMonth').CustomerID.count()/tx_min_purchase.groupby('SignupYearMonth').CustomerID.count()
tx_activation = tx_activation.reset_index()


# In[74]:


plot_data = [
    go.Bar(
        x=tx_activation.query("SignupYearMonth>201101 and SignupYearMonth<201109")['SignupYearMonth'],
        y=tx_activation.query("SignupYearMonth>201101 and SignupYearMonth<201109")['CustomerID'],
    )
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Activation Rate'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[75]:


tx_activation_ch = tx_min_purchase[tx_min_purchase['MinPurchaseYearMonth'] == tx_min_purchase['SignupYearMonth']].groupby(['SignupYearMonth','AcqChannel']).CustomerID.count()/tx_min_purchase.groupby(['SignupYearMonth','AcqChannel']).CustomerID.count()
tx_activation_ch = tx_activation_ch.reset_index()


# In[76]:


plot_data = [
    go.Scatter(
        x=tx_activation_ch.query("SignupYearMonth>201101 and SignupYearMonth<201108 and AcqChannel == 'organic'")['SignupYearMonth'],
        y=tx_activation_ch.query("SignupYearMonth>201101 and SignupYearMonth<201108 and AcqChannel == 'organic'")['CustomerID'],
        name="organic"
    ),
    go.Scatter(
        x=tx_activation_ch.query("SignupYearMonth>201101 and SignupYearMonth<201108 and AcqChannel == 'inorganic'")['SignupYearMonth'],
        y=tx_activation_ch.query("SignupYearMonth>201101 and SignupYearMonth<201108 and AcqChannel == 'inorganic'")['CustomerID'],
        name="inorganic"
    ),
    go.Scatter(
        x=tx_activation_ch.query("SignupYearMonth>201101 and SignupYearMonth<201108 and AcqChannel == 'referral'")['SignupYearMonth'],
        y=tx_activation_ch.query("SignupYearMonth>201101 and SignupYearMonth<201108 and AcqChannel == 'referral'")['CustomerID'],
        name="referral"
    )
    
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Activation Rate - Channel Based'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# # Monthly Retention Rate

# In[77]:


tx_uk.head()


# In[78]:


df_monthly_active = tx_uk.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()


# In[79]:


tx_user_purchase = tx_uk.groupby(['CustomerID','InvoiceYearMonth'])['Revenue'].sum().astype(int).reset_index()


# In[80]:


tx_user_purchase


# In[81]:


tx_user_purchase.Revenue.sum()


# In[82]:


tx_retention = pd.crosstab(tx_user_purchase['CustomerID'], tx_user_purchase['InvoiceYearMonth']).reset_index()


# In[83]:


tx_retention.head()


# In[84]:


months = tx_retention.columns[2:]


# In[85]:


months


# In[86]:


retention_array = []
for i in range(len(months)-1):
    retention_data = {}
    selected_month = months[i+1]
    prev_month = months[i]
    retention_data['InvoiceYearMonth'] = int(selected_month)
    retention_data['TotalUserCount'] = tx_retention[selected_month].sum()
    retention_data['RetainedUserCount'] = tx_retention[(tx_retention[selected_month]>0) & (tx_retention[prev_month]>0)][selected_month].sum()
    retention_array.append(retention_data)
    


# In[87]:


tx_retention = pd.DataFrame(retention_array)


# In[88]:


tx_retention.head()


# In[89]:


tx_retention['RetentionRate'] = tx_retention['RetainedUserCount']/tx_retention['TotalUserCount']


# In[90]:


tx_retention


# In[91]:


plot_data = [
    go.Scatter(
        x=tx_retention.query("InvoiceYearMonth<201112")['InvoiceYearMonth'],
        y=tx_retention.query("InvoiceYearMonth<201112")['RetentionRate'],
        name="organic"
    )
    
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Retention Rate'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# # Churn Rate

# In[92]:


tx_retention['ChurnRate'] =  1- tx_retention['RetentionRate']


# In[93]:


plot_data = [
    go.Scatter(
        x=tx_retention.query("InvoiceYearMonth<201112")['InvoiceYearMonth'],
        y=tx_retention.query("InvoiceYearMonth<201112")['ChurnRate'],
        name="organic"
    )
    
]

plot_layout = go.Layout(
        xaxis={"type": "category"},
        title='Monthly Churn Rate'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# # Cohort Base Retention

# In[94]:


tx_user_purchase.head()


# In[95]:


tx_min_purchase.head()


# In[96]:


tx_retention = pd.crosstab(tx_user_purchase['CustomerID'], tx_user_purchase['InvoiceYearMonth']).reset_index()


# In[97]:


tx_retention = pd.merge(tx_retention,tx_min_purchase[['CustomerID','MinPurchaseYearMonth']],on='CustomerID')


# In[98]:


tx_retention.head()


# In[99]:


tx_retention.columns


# In[100]:


new_column_names = [ 'm_' + str(column) for column in tx_retention.columns[:-1]]
new_column_names.append('MinPurchaseYearMonth')


# In[101]:


tx_retention.columns = new_column_names


# In[102]:


tx_retention


# In[103]:


months


# In[114]:


retention_array = []
for i in range(len(months)):
    retention_data = {}
    selected_month = months[i]
    prev_months = months[:i]
    next_months = months[i+1:]
    for prev_month in prev_months:
        retention_data[prev_month] = np.nan
        
    total_user_count = tx_retention[tx_retention.MinPurchaseYearMonth ==  selected_month].MinPurchaseYearMonth.count()
    retention_data['TotalUserCount'] = total_user_count
    retention_data[selected_month] = 1 
    
    query = "MinPurchaseYearMonth == {}".format(selected_month)
    

    for next_month in next_months:
        new_query = query + " and {} > 0".format(str('m_' + str(next_month)))
        retention_data[next_month] = np.round(tx_retention.query(new_query)['m_' + str(next_month)].sum()/total_user_count,2)
    retention_array.append(retention_data)


# In[107]:


tx_retention = pd.DataFrame(retention_array)


# In[108]:


len(months)


# In[109]:


tx_retention.index = months


# In[110]:


tx_retention


# In[ ]:




