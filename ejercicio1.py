# regresion lineal
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Hemos  creado un csv con los datos que se van a graficar
datos = pd.DataFrame({'A': [1,2,4,7,16,32,64], 'S': [2,4,7,11,16,19,21]})
print(datos)

#a) Grafica de nube de puntos

plt.scatter(datos['A'],datos['S'])
plt.plot(np.unique(datos['A']), np.poly1d(np.polyfit(datos['A'], datos['S'], 1))(np.unique(datos['A'])))
plt.show()


X = datos[['A']]   
y = datos['S']

regr = linear_model.LinearRegression()
regr.fit(X, y)

y_pred = regr.predict(X)


# PARA UNA A DE 100 CUANTO VALE S
print(regr.predict([[100]]))

# covarianza de xy
cov = np.cov(datos['A'],datos['S'])
print(cov)

#  varianza de x y y
var = np.var(datos['A'])
print(var)

var = np.var(datos['S'])
print(var)




