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
[Sklearn's train_test_split class](http://scikit-learn.org/stable/modules/cross_validation.html) is used to split the data into train and test sets.

### Feature Scaling
If we consider the above same data shown in Encoding Categorical Variables section, the values of age is much smaller than the values of salary. So in such cases, Models can create a system assuming that the output is more dependent on salary than on age. So to avoid such cases, we use [Sklearn's StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) to scale the values cantered around 0.  

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
X[:,:] = labelEncoder_X.fit_transform(X[:,:])
oneHotEncoder_X = OneHotEncoder(categorical_features = [*****Index of Categorical Column*****])
X = oneHotEncoder_X.fit_transform(X).toarray()

#splitting the data set to Training and Test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # test_size is the percentage of test set size (20% in this case)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
```

