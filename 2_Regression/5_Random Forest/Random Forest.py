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