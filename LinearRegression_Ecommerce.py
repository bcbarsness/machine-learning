
#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Get the Data
#
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 

customers = pd.read_csv("EcommerceCustomers.csv")


print(customers.head())

print(customers.describe())

print(customers.info())

# ## Compare Website vs App on impact to Yearly Spending
sns.set_palette("GnBu_d")
#sns.set_style('whitegrid')
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
sns.plt.show()

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
sns.plt.show()

# ** Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)
sns.plt.show()

# ** Explore relationships across the entire data set. 

sns.pairplot(customers)
sns.plt.show()

# **Create a linear model plot of  Yearly Amount Spent vs. Length of Membership. **

sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
sns.plt.show()

# ## Training and Testing Data
# 
# Split the data into training and testing sets.

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


# ** Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Training the Model
 
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

# The coefficients

print('Coefficients: \n', lm.coef_)

# ## Predicting Test Data

predictions = lm.predict( X_test)


# ** Create a scatterplot of the real test values versus the predicted values. **

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()

# ## Evaluating the Model
# 

# ** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ## Residuals
# 
# Explore the residuals to make sure everything was okay with our data. 
# 
# **Plot a histogram of the residuals and make sure it looks normally distributed. 

sns.distplot((y_test-predictions),bins=50);


# ## Conclusion
# We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see if we can interpret the coefficients at all to get an idea.

# Interpreting the coefficients:
# 
# - Holding all other features fixed, a 1 unit increase in **Avg. Session Length** is associated with an **increase of 25.98 total dollars spent**.
# - Holding all other features fixed, a 1 unit increase in **Time on App** is associated with an **increase of 38.59 total dollars spent**.
# - Holding all other features fixed, a 1 unit increase in **Time on Website** is associated with an **increase of 0.19 total dollars spent**.
# - Holding all other features fixed, a 1 unit increase in **Length of Membership** is associated with an **increase of 61.27 total dollars spent**.

