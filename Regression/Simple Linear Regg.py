# Simple Linear Regg

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

#Diving Independent and Dependent parameters
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#Training the Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Testing the model
Y_pred = regressor.predict(X_test)

#Visulalising Data Model
plt.scatter(X_test,Y_test,color = 'red')
plt.plot(X_test,regressor.predict(X_test), color = 'blue')
plt.title('My 1st Model - Sal vs YearsOfExp')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.show()
