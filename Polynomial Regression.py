import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

"""from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=1/3,random_state=0)"""

"""#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
sc_y=StandardScaler()
y_train=sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
regressor2=LinearRegression()
regressor2.fit(X_poly,y)

plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

