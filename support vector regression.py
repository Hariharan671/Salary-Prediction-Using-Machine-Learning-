import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:3].values

"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X= LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()"""

"""from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=1/3,random_state=0)"""

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
sc_y= StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor =SVR(kernel='rbf')
regressor.fit(X,y)

y_pred =sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff(Support vector Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
