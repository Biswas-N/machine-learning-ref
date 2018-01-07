# Data PreProcessing

This is an very important step to consider to get optimal results from a machine learning model.

## Topics
- Importing Datasets
- Missing Data Handelling
- Encoding Categorical Variables
- Splitting Data to Train and Test
- Feature Scaling

### Importing Datasets
Pandas class is used to import data. This is a list of [Data Formats](https://pandas.pydata.org/pandas-docs/stable/io.html) which can be accessed using Pandas. The datasets imported by pandas are stored in the form of Pandas DataFrame Type. To access data in such DataFrames we need to use [**iloc**](http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.DataFrame.iloc.html).

### Handelling Missing data
[Sklearns's Imputer Class](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) is used for indentifying the missing data and Strategically handelling the missing data.

### Encoding Categorical Variables
To best explain this situation lets take some example data:

![data](https://user-images.githubusercontent.com/24390015/34645858-91c8a616-f3ac-11e7-9aa9-60235a5f6ad1.JPG)
###### Data Source: Udemy
There are two categorical values: Country and Purchased. As machine learning models are mathematical models, we might get some unexpected results if we use such categorical variable. Thus we use Encoders to convert such variables into numbers. So we Use two encoders: [**LabelEncoder**](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html) and [**OneHotEncoder**](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

### Splitting Data


## Code
```py
# Data Preprocessing

# Importing the libraries
import numpy as np # Has mathematical tools to perform Array Addition, Multiplication etc.
import matplotlib.pyplot as plt # Used for plotting the Models outputs as Graphs
import pandas as pd # Access and manage datasets

# Importing the dataset
dataset = pd.read('*****Your Datasets*****') # file can be of various formats
X = dataset.iloc[:, :].values # iloc stands for Integer Location
y = dataset.iloc[:, :].values # syntax: iloc[rows,columns] and start(inclusive):End(Exclusive)

#fixing mixed data
from sklearn.preprocessing import Imputer # Imputer class contains methods to handle 
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0) # See documentation for Attributes
imputer = imputer.fit(X[:,:]) # Dataset containing missng data must be given
X[:,:] = imputer.fit_transform(X[:,:])

#Encoding Values of X
from sklearn.preprocessing import OneHotEncoder, LabelEncoder # Classes for Encoding data
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
oneHotEncoder_X = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder_X.fit_transform(X).toarray()
```

