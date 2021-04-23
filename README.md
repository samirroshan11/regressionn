import pandas as pd # install pandas
# pip install scipy - actually provides you with the statistical prediction model
from scipy import stats
from sklearn.linear_model import LinearRegression # Install sckit learn
import matplotlib.pyplot as plt 
nyc = pd.read_csv(r'c:\Users\mousu\climate.csv' ,sep = ',' , na_values = ['no info', ','] )
print(nyc)
print(nyc.head())
print(nyc.tail())
# Change the column names

nyc.columns = ['Date', 'Temperature', 'Anamoly']
print(nyc)
print(nyc.Date.dtype)
# Calling series method Floordiv performs integer division on every element of the series

nyc.Date = nyc.Date.floordiv(100) # It is changes the format of the data, this is optional 
print(nyc)
# Forcasting Future March Average  temparature
linear_regression = stats.linregress(x=nyc.Date, y = nyc.Temperature)# This package is coming from your sckit learn 
slope =  linear_regression.slope # Package from scipy
print(slope)
intercept = linear_regression.intercept
print(intercept)
# Predicting the March temperature in year 2050
predict = linear_regression.slope *2050 + linear_regression.intercept
print(predict)
# Predicting the March temp in year 1892
predict_1 = linear_regression.slope *1892+ linear_regression.intercept
print(predict_1)

year = 2020
while predict <46.0 or predict == 46.0: # A temperature which is less that < or equal to 46
           year += 1
           predict = slope*year + intercept
print(year)
X = nyc.iloc[:, 0].values.reshape(-1, 1)  # This is getting the columns by pandas and shaping the columns 
Y = nyc.iloc[:, 1].values.reshape(-1, 1)  # This is getting the columns by pandas and shaping the columns
linear_regressor = LinearRegression()  # create object for the class linear_regressor class we are creating and that's going to fit our data
linear_regressor.fit(X, Y)  # perform linear regression-fitting the line
Y_pred = linear_regressor.predict(X)  # make predictions We are predicting Y, given X.
plt.scatter(X, Y, color = 'blue') # Matplotlib
plt.plot(X, Y_pred, color='orange', linewidth = 7) # That's going to create the prediction line
plt.show()
