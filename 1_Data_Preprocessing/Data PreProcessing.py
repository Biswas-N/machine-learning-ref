# Data Preprocessing

# Importing the libraries
import numpy as np # Has mathematical tools to perform Array Addition, Multiplication etc.
import matplotlib.pyplot as plt # Used for plotting the Models outputs as Graphs
import pandas as pd # Access and manage datasets

# Importing the dataset
dataset = pd.read_csv('Data.csv') # file can be of various [formats](https://pandas.pydata.org/pandas-docs/stable/io.html)
X = dataset.iloc[:, :-1].values # iloc stands for Integer Location
y = dataset.iloc[:, 3].values # iloc[rows,columns] and start(inclusive):End(Exclusive)

#fixing mixed data
from sklearn.preprocessing import Imputer # Imputer class contains methods to handle 
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.fit_transform(X[:,1:3])

#Encoding Values of X
from sklearn.preprocessing import OneHotEncoder, LabelEncoder # Classes for Encoding data
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
oneHotEncoder_X = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder_X.fit_transform(X).toarray()

#encoding values of Y
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

#splitting the data set to Training and Test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)