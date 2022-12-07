#!/usr/bin/env python
# coding: utf-8

# In[183]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('pylab', 'inline')
import seaborn as sns

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

titanic = pd.read_csv('/Users/hetvikavi/Desktop/STAT 1P50/titanic3.csv')

titanic.info()

titanic['Survival'] = titanic.boat.map({0 : 'Died', 1 : 'Survived'})
titanic.boat.head()


male_female_survival = titanic.groupby('sex').sum()['survived']
print(male_female_survival)
total_male_female = titanic['sex'].value_counts()
print(total_male_female)

male_female_percentages = male_female_survival/total_male_female
print(male_female_percentages)

male_female_percentages = male_female_survival/total_male_female
print(male_female_percentages)
prob_gender = male_female_percentages.plot(kind='barh',title='Chance of Survival per Gender')
prob_gender.set_xlabel('Probability');
prob_gender.set_ylabel('Sex');



# In[317]:


survived = data[data['survived']==1]
not_survided = data[data['survived']==0]
total_passenger = len(data.survived)
print("\033[1m" + 'What was the percentage of survived?' + "\033[0m")
print('-'*10)
print('The percentage of survived was:{:.2f}%'.format((len(survived)/total_passenger)*100))
print('The percentage of not survived was:{:.2f}%'.format((len(not_survided)/total_passenger)*100))
print('-'*10)

woman = data[data['sex']=='female'].dropna(axis=1)
men = data[data['sex']=='male'].dropna(axis=1)

print("\033[1m" +'How many male and female embarked?' + "\033[0m")
print('-'*10)
print('Total of female embarked on Titanic was {}'.format(len(woman)))
print('Total of male embarked on Titanic was {}'.format(len(men)))


# In[341]:



import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


data = pd.read_csv('/Users/hetvikavi/Desktop/STAT 1P50/titanic3.csv')

s = pd.get_dummies(df['sex'])

X, y = data.drop(['survived'],axis=1), data[['survived']]

feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = LogisticRegression()
model = model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[340]:


import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data = pd.read_csv('/Users/hetvikavi/Desktop/STAT 1P50/titanic3.csv')

s= pd.get_dummies(df['sex'])
s.drop(['male'],axis=1)
data['sex']=s['female']
data['sex'].nunique()

data.head()
data.tail()

X = data.drop(['survived'],axis=1)
y = data['survived']

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=100)
X.head()
print(X_train.shape)
print(y_train.shape)

model = tree.DecisionTreeClassifier(random_state = 0)
model.fit(X_train, y_train)
tree.plot_tree(model)

y_pred_train = model.predict(X_train)  
y_pred_test = model.predict(X_test)

print('Accuracy score for test data is:', accuracy_score(y_test,y_pred_test))

