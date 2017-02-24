#!/usr/bin/python3 

# The data contains the following columns:
# 
# * 'Avg. Area Income': Avg. Income of residents of the city house is located in.
# * 'Avg. Area House Age': Avg Age of Houses in same city
# * 'Avg. Area Number of Rooms': Avg Number of Rooms for Houses in same city
# * 'Avg. Area Number of Bedrooms': Avg Number of Bedrooms for Houses in same city
# * 'Area Population': Population of city house is located in
# * 'Price': Price that the house sold at
# * 'Address': Address for the house

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

USAhousing = pd.read_csv('USA_Housing.csv')

print(USAhousing.head())

print(USAhousing.info())

print(USAhousing.describe())

print(USAhousing.columns)


sns.pairplot(USAhousing)
sns.plt.show()

sns.distplot(USAhousing['Price'])
sns.plt.show()

sns.heatmap(USAhousing.corr(), annot=True)
sns.plt.show()

#######################################
# ## Training a Linear Regression Model
# 
# First split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the Price column. 
# 

X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']


# ## Train Test Split
# 
# Split the data into a training set and a testing set. 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# ## Creating and Training the Model

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# ## Model Evaluation
# 
# Evaluate the model by checking out it's coefficients and how we can interpret them.

# print the intercept
print(lm.intercept_)

# print the coefficient
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print(coeff_df)


# Interpreting the coefficients:  
# 
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Income** is associated with an **increase of \$21.52 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area House Age** is associated with an **increase of \$164883.28 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Rooms** is associated with an **increase of \$122368.67 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Bedrooms** is associated with an **increase of \$2233.80 **.
# - Holding all other features fixed, a 1 unit increase in **Area Population** is associated with an **increase of \$15.15 **.
# 

# ## Predictions from our Model
# 
# Get predictions off our test set and plot

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.show()

# **Residual Histogram**

sns.distplot((y_test-predictions),bins=50);
sns.plt.show()

# ## Regression Evaluation Metrics
# 
# 
# Here are three common evaluation metrics for regression problems:
# 
# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# 
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# **Mean Squared Error** (MSE) is the mean of the squared errors:
# 
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# 
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
# 
# Comparing these metrics:
# 
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
# 

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


