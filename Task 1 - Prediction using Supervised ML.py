#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION

# # Data Science and Business Analytics  

# ## Task-1 : Prediction Using Supervised ML

# ### By- ADITYA SINGH

# ### Question: What will we the Predicted score if a student studies for 9.25 hour/day?

# ##### Importing Libraries

# In[32]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt


# ##### Loading the Data

# In[34]:


data = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")


# In[87]:


data


# In[88]:


data.shape


# In[89]:


data.describe()


# In[90]:


#checking the column names
df.columns


# ##### Visualising the Variables

# In[91]:


plt.scatter(x = data.Hours, y = data.Scores, color = 'b', alpha = 0.75)
plt.title("Hours vs Scores")
plt.xlabel("Study Hours")
plt.ylabel("Student Scores")


# ###### Hence, Scores are linearly dependent on Hours

# In[92]:


import seaborn as sns

sns.heatmap(data.corr(), annot= True)


# ###### Splitting into Training Set and Test Set

# In[93]:


X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values


# In[94]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 5)


# ###### Simple Linear Regression model on the Training set

# In[95]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# ###### Plotting the regression line and Test Data

# In[96]:


line = regressor.coef_*X+regressor.intercept_

plt.scatter(X, Y, color = 'r')
plt.plot(X, line,color = 'g');
plt.show()


# ###### Predicting the Test Set result

# In[86]:


Y_pred = regressor.predict(X_test)
print(Y_pred)


# ###### Visualising the Training Set results

# In[97]:


plt.scatter(X_train, Y_train, color = 'g')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Hours vs. Percentage (Training set)')
plt.xlabel('Hours studied')
plt.ylabel('Percentage of marks')
plt.show()


# ###### Predicting the Score

# In[99]:


data = np.array(9.25)
data = data.reshape(-1, 1)
pred = regressor.predict(data)
print("If the student studies for 9.25 hours/day, the score is {}.".format(pred))


# ###### Conclusion:

# We used a Linear Regression Model to predict the score of a student if he/she studies for 9.25 hours/day and the Predicted Score came out to be 90.73666515.
