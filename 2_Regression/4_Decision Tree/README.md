# Decision Tree regression
 This algorithm is of both Regression and Classification, But we will be talking about Regression. It can predict Dependent variables based on given Independent variables.
	
## Explanation:
Theoretically the prediction is done by splitting the Scatter Plots of the Independent variables (X1 and X2) and computing the value of Y by averaging all the values of Y corresponding to X1 and X2 in their split regions (As Shown below)
 
  
![decision tree intuition](https://user-images.githubusercontent.com/24390015/34588791-fa9f7f8a-f202-11e7-9ed5-7ef608a43bbd.jpg)
###### Image Source: Udemy
  
Later these scatter plot splits are constructed into decision trees.
![decision tree intuition 2](https://user-images.githubusercontent.com/24390015/34588825-1ff55b38-f203-11e7-838a-3d1ad0d51b09.JPG)
###### Image Source: Udemy


# Lets Do It!   
## Dataset Used
![decision tree dataset](https://user-images.githubusercontent.com/24390015/34589991-1b7d0736-f208-11e7-9dbe-c9154ead724d.JPG)
###### Data Source: Udemy
- Independent Variable: Level
- Dependent Variable: Salary

# Code
```py
# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset and Splitting them into Independent and Dependent Variables
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Building the Descision Tree Model
#Creation
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state = 0)
#Fitting
reg.fit(X,y)
#Testing or Predicting
y_pred = reg.predict(6.5)

#Plotting the Regressor performance
#For High Accurate graph
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
#Creating plt
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
```

## Model Output
![x](https://user-images.githubusercontent.com/24390015/34589917-bcd1ad04-f207-11e7-8f31-8357d876581d.png) 

As we are using only one independent variable in this example we can see that the splits took place only along X-axis.
