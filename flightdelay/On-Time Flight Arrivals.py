#!/usr/bin/env python
# coding: utf-8

# In[49]:


get_ipython().system(u'curl https://topcs.blob.core.windows.net/public/FlightData.csv -o flightdata.csv')


# In[50]:


import pandas as pd

df = pd.read_csv('flightdata.csv')
df.head()


# In[51]:


df.shape


# In[52]:


df.isnull().values.any()


# In[53]:


df.isnull().sum()


# In[54]:


df = df.drop('Unnamed: 25', axis = 1)
df.isnull().sum()


# In[55]:


features = ["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_DEP_TIME", "ARR_DEL15"]
df = df[features]
df.isnull().sum()


# In[56]:


df.columns


# In[57]:


df[df.isnull().values.any(axis=1)].head()


# In[58]:


df = df.fillna({'ARR_DEL15': 1})
df.iloc[177:185]


# In[ ]:





# In[59]:


df.isnull().sum()


# In[60]:


import math

for index, row in df.iterrows():
    df.loc[index, 'CRS_DEP_TIME'] = math.floor(row['CRS_DEP_TIME'] / 100)
df.head()


# In[61]:


df = pd.get_dummies(df, columns=['ORIGIN', 'DEST'])
df.head()


# In[64]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df.drop('ARR_DEL15', axis=1), df['ARR_DEL15'], test_size=0.2, random_state=42)
train_x.shape


# In[65]:


test_x.shape


# In[66]:


train_y.shape


# In[67]:


test_y.shape


# In[68]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=13)
model.fit(train_x, train_y)


# In[69]:


predicted = model.predict(test_x)
model.score(test_x, test_y)


# In[70]:


from sklearn.metrics import roc_auc_score
probabilities = model.predict_proba(test_x)


# In[71]:


roc_auc_score(test_y, probabilities[:, 1])


# In[72]:


from sklearn.metrics import confusion_matrix
confusion_matrix(test_y, predicted)


# In[73]:


from sklearn.metrics import precision_score

train_predictions = model.predict(train_x)
precision_score(train_y, train_predictions)


# In[74]:


from sklearn.metrics import recall_score

recall_score(train_y, train_predictions)


# In[75]:


get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[76]:


from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(test_y, probabilities[:, 1])
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')


# In[78]:


def predict_delay(departure_date_time, origin, destination):
    from datetime import datetime

    try:
        departure_date_time_parsed = datetime.strptime(departure_date_time, '%d/%m/%Y %H:%M:%S')
    except ValueError as e:
        return 'Error parsing date/time - {}'.format(e)

    month = departure_date_time_parsed.month
    day = departure_date_time_parsed.day
    day_of_week = departure_date_time_parsed.isoweekday()
    hour = departure_date_time_parsed.hour

    origin = origin.upper()
    destination = destination.upper()

    input = [{'MONTH': month,
              'DAY': day,
              'DAY_OF_WEEK': day_of_week,
              'CRS_DEP_TIME': hour,
              'ORIGIN_ATL': 1 if origin == 'ATL' else 0,
              'ORIGIN_DTW': 1 if origin == 'DTW' else 0,
              'ORIGIN_JFK': 1 if origin == 'JFK' else 0,
              'ORIGIN_MSP': 1 if origin == 'MSP' else 0,
              'ORIGIN_SEA': 1 if origin == 'SEA' else 0,
              'DEST_ATL': 1 if destination == 'ATL' else 0,
              'DEST_DTW': 1 if destination == 'DTW' else 0,
              'DEST_JFK': 1 if destination == 'JFK' else 0,
              'DEST_MSP': 1 if destination == 'MSP' else 0,
              'DEST_SEA': 1 if destination == 'SEA' else 0 }]

    return model.predict_proba(pd.DataFrame(input))[0][0]


# In[79]:


predict_delay('1/10/2018 21:45:00', 'JFK', 'ATL')


# In[80]:


predict_delay('2/10/2018 21:45:00', 'JFK', 'ATL')


# In[81]:


predict_delay('2/10/2018 10:00:00', 'ATL', 'SEA')


# In[82]:


import numpy as np

labels = ('Oct 1', 'Oct 2', 'Oct 3', 'Oct 4', 'Oct 5', 'Oct 6', 'Oct 7')
values = (predict_delay('1/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('2/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('3/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('4/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('5/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('6/10/2018 21:45:00', 'JFK', 'ATL'),
          predict_delay('7/10/2018 21:45:00', 'JFK', 'ATL'))
alabels = np.arange(len(labels))

plt.bar(alabels, values, align='center', alpha=0.5)
plt.xticks(alabels, labels)
plt.ylabel('Probability of On-Time Arrival')
plt.ylim((0.0, 1.0))


# In[84]:


import numpy as np

labels = ('9:00am', 'Noon', '3:00pm', '6:00pm', '9:00pm')
values = (predict_delay('30/01/2018 09:00:00', 'SEA', 'ATL'),
          predict_delay('30/01/2018 12:00:00', 'SEA', 'ATL'),
          predict_delay('30/01/2018 15:00:00', 'SEA', 'ATL'),
          predict_delay('30/01/2018 18:00:00', 'SEA', 'ATL'),
          predict_delay('30/01/2018 21:00:00', 'SEA', 'ATL'))
        
alabels = np.arange(len(labels))

plt.bar(alabels, values, align='center', alpha=0.5)
plt.xticks(alabels, labels)
plt.ylabel('Probability of On-Time Arrival')
plt.ylim((0.0, 1.0))


# In[ ]:




