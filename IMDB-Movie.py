#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv('IMDB-Movie-Data.csv')


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


print("Number of rows",data.shape[0])
print("Number of columns",data.shape[1])


# In[7]:


data.info()


# In[8]:


data.isnull().sum()


# In[9]:


sns.heatmap(data.isnull())


# In[10]:


per_missing=data.isnull().sum()*100/len(data)
per_missing


# In[13]:


data.dropna(axis=0)


# In[14]:


dup_data=data.duplicated().any()
print("Are there any duplicate values?",dup_data)


# In[15]:


data=data.drop_duplicates()
data


# In[19]:


data.describe(include="all") # get overall statistics of data


# In[20]:


data.columns


# ## Display Title of the movie having runtime>=180 minutes

# In[22]:


data[data['Runtime (Minutes)']>=180]['Title']


# ## In which year there was the highest average voting?

# In[23]:


data.groupby('Year')['Votes'].mean().sort_values(ascending=False)


# In[26]:


sns.barplot(x='Year',y='Votes',data=data)
plt.title("Votes by Year")
plt.show()


# # # In which year there was the highest average revenue

# In[27]:


data.groupby('Year')['Revenue (Millions)'].mean().sort_values(ascending=False)


# In[28]:


sns.barplot(x='Year',y='Revenue (Millions)',data=data)
plt.title("Votes by Year")
plt.show()


# ## Find the average rating for each director

# In[29]:


data.columns


# In[30]:


data.groupby('Director')['Rating'].mean().sort_values(ascending=False)


# # #Display Top 10 lengthy movies title and runtime

# In[34]:


top10_len=data.nlargest(10,'Runtime (Minutes)')[['Title','Runtime (Minutes)']].set_index('Title')


# In[35]:


top10_len


# In[36]:


sns.barplot(x='Runtime (Minutes)',y=top10_len.index,data=top10_len)


# In[ ]:


##Display Number of movies per year


# In[37]:


data['Year'].value_counts()


# In[39]:


sns.countplot(x='Year',data=data)
plt.title("Number of Movies per year")
plt.show()


# # #Find most popular movie title(Highest Revenue)

# In[43]:


data[data['Revenue (Millions)'].max()==data['Revenue (Millions)']]['Title']


# # Display top 10 highest rated movie titles and its directors

# In[44]:


data.columns


# In[45]:


top10_len=data.nlargest(10,'Rating')[['Title','Rating','Director']].set_index('Title')


# In[46]:


top10_len


# In[49]:


sns.barplot(x='Rating',y=top10_len.index,data=top10_len,hue='Director',dodge=False)

plt.legend(bbox_to_anchor=(1.05,1),loc=2)


# ## Display Top 10 Highest Revenue Movie Titles

# In[50]:


data.nlargest(10,'Revenue (Millions)')['Title']


# In[54]:


top10=data.nlargest(10,'Revenue (Millions)')[['Title','Revenue (Millions)']].set_index('Title')


# In[55]:


top10


# In[60]:


sns.barplot(x='Revenue (Millions)',y=top10.index,data=top10)
plt.title('Top 10 Highest Movie Title')
plt.show()


# # #Find Average Rating of Movies Year Wise

# In[61]:


data.groupby('Year')['Rating'].mean().sort_values(ascending=False)


# In[ ]:


#Does rating affect the revenue


# In[62]:


sns.scatterplot(x='Rating',y='Revenue (Millions)',data=data)


# ## Classify Movies Based on Ratings [Excellent, Good, and Average]

# In[63]:


def rating(rating):
    if rating>=7.0:
        return "Excellent"
    elif  rating>=6.0:
        return "Good"
    else:
        return "Average"


# In[65]:


data['rating_cat']=data['Rating'].apply(rating)


# In[66]:


data.head()


# ## Count Number of Action Movies

# In[67]:


data['Genre'].dtype


# In[68]:


len(data[data['Genre'].str.contains('Action',case=False)])


# # Find Unique Values From Genre 

# In[69]:


data['Genre']


# In[70]:


list1=[]
for value in data['Genre']:
    list1.append(value.split(','))


# In[71]:


list1


# In[72]:


one_d=[]
for item in list1:
    for item1 in item:
        one_d.append(item1)


# In[73]:


one_d


# In[82]:


uni_list=[]
for item in one_d:
    if item not in uni_list:
        uni_list.append(item)


# In[84]:


len(uni_list)


# # How Many Films of Each Genre Were Made?

# In[87]:


one_d=[]
for item in list1:
    for item1 in item:
        one_d.append(item1)


# In[88]:


from collections import Counter


# In[90]:


Counter(one_d)


# In[ ]:




