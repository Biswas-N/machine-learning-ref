# Random Forest regression
 This algorithm is a version (or) type of [**Ensemble Learning**](https://en.wikipedia.org/wiki/Ensemble_learning).
	
## Explanation:
This algorithm combines **multiple number of Decision Tree (Forest)** Regression models by averaging the outputs of all the decision trees. So theoretically the prediction is done by averaging all the individual outputs or prediction of the trees in the forest.

The forest can have any number of Decision Trees, this can be changed by changing the value of "n_estimators" in RandomForestRegressor Class object(Show in detail below).

# Lets Do It!   
## Dataset Used
![decision tree dataset](https://user-images.githubusercontent.com/24390015/34589991-1b7d0736-f208-11e7-9dbe-c9154ead724d.JPG)
###### Data Source: Udemy
- Independent Variable: Level
- Dependent Variable: Salary

# Code
```py
# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting the Random Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators = 300, random_state = 0) # For 10 trees change n_estimators to 10
reg.fit(X, y)

# Predicting a new result
y_pred = reg.predict(6.5)

# Visualising the Random Forest Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, reg.predict(X_grid), color = 'blue')
plt.title('Random Forest Model (300 Decision Trees)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
```

## Model Output
### For 10 Decision Trees in forest
![for 10](https://user-images.githubusercontent.com/24390015/34645525-0fb8e22e-f3a4-11e7-8495-214ce5947154.png)

### For 300 Decision Trees in forest
![for 300](https://user-images.githubusercontent.com/24390015/34645527-10164252-f3a4-11e7-914d-7db9b3e9cd76.png)

We can see that there is no change in the Model Performance by increasing the number of Trees in the forest and Similarly, As we are using only one independent variable (As shown in the example of Decision tree) in this example we can see that the splits took place only along X-axis.
