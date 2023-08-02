# Import necessary libraries
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import Ridge, Lasso
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import cross_val_score

# Day 1 and Day 2
np.random.seed(42)
X1 = np.random.rand(100)
epsilon = np.random.normal(0, 1, 100) # Noise term
y = 3 * X1 + 2 + epsilon # Adding noise to the linear relationship

# Day 3
X = np.column_stack((np.ones(100), X1))
beta = np.linalg.inv(X.T @ X) @ X.T @ y
print("Beta values: ", beta)

# Day 4
model = sm.OLS(y, X).fit()
print(model.summary())

# Day 5
X2 = np.random.rand(100)
y = 3 * X1 + 2 * X2 + 2 + epsilon # Adding another independent variable X2
X = np.column_stack((np.ones(100), X1, X2)) # Adding intercept and combining X1 and X2
model = sm.OLS(y, X).fit()

# Day 6
plt.scatter(model.predict(X), model.resid)
plt.title('Residual Plot')
plt.show()

# Day 7
y_pred = model.predict(X)
print('RMSE:', np.sqrt(mean_squared_error(y, y_pred)))
print('MAE:', mean_absolute_error(y, y_pred))
print('R-squared:', r2_score(y, y_pred))

# Day 8-9 skipped for brevity, implement transformations as needed

# Day 10
print('Durbin-Watson statistic:', durbin_watson(model.resid))

# Day 11
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# Day 12-13
sfs = SFS(lasso, 
          k_features=3, 
          forward=True, 
          floating=False, 
          scoring='r2',
          cv=5)
sfs = sfs.fit(X, y)

print('Cross-validation score:', cross_val_score(lasso, X, y, cv=5).mean())
