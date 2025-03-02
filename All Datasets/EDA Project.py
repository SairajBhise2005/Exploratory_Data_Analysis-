#!/usr/bin/env python
# coding: utf-8

# In[22]:


pip


# In[23]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install matplot')
get_ipython().system('pip install numpy')


# In[24]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[39]:


medallists = pd.read_csv("C:\\Users\\priya\\Downloads\\medallists.csv")
df1=pd.DataFrame(medallists)
athletes = pd.read_csv("C:\\Users\\priya\\Downloads\\athletes.csv")
df2 =pd.DataFrame(athletes)
df1
total_medal = pd.read_csv("C:\\Users\\priya\\Downloads\\medals_total.csv")
df_medals_total = pd.DataFrame(total_medal)


# # #Merging the datasets

# In[31]:


merged = pd.merge(df1,df2,on='name',how='inner')
merged


# # Cleaning the data

# In[35]:


merged.fillna(0, inplace=True)
merged.drop_duplicates(keep='first', inplace=True)
merged


# 
# ## Data Type Classification

# In[ ]:


numerical = merged.select_dtypes(include=['int64','float64']).columns
print("Numerical Columns:",numerical.tolist())
num = numerical.tolist()
categorical = merged.select_dtypes(include=['object']).columns
print("\nCategorical Columns: ",categorical.tolist())
cat = categorical.tolist()


# In[ ]:


def check_discrete_or_continuous(df, num):
    for var in num:
        if all(df[var].apply(lambda x: float(x).is_integer())):
            print(f"Column '{var}' is discrete.")
        else:
            print(f"Column '{var}' is continuous.")
check_discrete_or_continuous(merged,num)


# In[ ]:


def classify_nominal_ordinal(df,categorical):
   ordinal_keywords =['level','rank','grade','stage','position','order','age','class','medal_type']
   nominal = []
   ordinal = []
   for col in categorical:
       if any(keyword in col.lower() for keyword in ordinal_keywords):
           ordinal.append(col)
       else:
           nominal.append(col)
       return nominal,ordinal
nominal, ordinal = classify_nominal_ordinal(merged,categorical)
print("Nominal Columns:",nominal)
print("Ordinal Columns:",ordinal)


# In[38]:


plt.figure(figsize=(10,5))
plt.bar(merged.index, merged['medal_code'],color='b')
plt.title('Monthly Average Temperature - Bar Plot')
plt.xlabel('medal_code')
plt.ylabel('les')
plt.xticks(rotation=45)
plt.show()


# In[42]:


import plotly.express as px
fig = px.bar(df_medals_total, x='country_code', y=['Gold Medal', 'Silver Medal', 'Bronze Medal'],
             title='Medal Counts by Country')
fig.update_layout(barmode='stack')
fig.show()


# In[46]:


palette = {
    'Gold Medal': 'gold',
    'Silver Medal': 'silver',
    'Bronze Medal': 'brown',
    'Male': 'blue',  
    'Female': 'pink'         
}

sns.set(style="whitegrid")
ax = sns.histplot(merged, x='medal_type', hue='gender_x', multiple="stack", palette=palette)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




