import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data_set = pd.read_csv('SAT_GPA_Data.csv')
x = data_set['GPA']
y = data_set['SAT']
x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
model = LinearRegression()
result = model.fit(X_train, y_train)
ypredict = model.predict(X_test)
plt.scatter(X_test, y_test, s=15)
plt.plot(X_test, ypredict, color='g')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print("No of records:", data_set.shape[0])
print("Mean: ", data_set.mean())
print("Standard Deviation: ", data_set.std())
print("Median: ", data_set.median())
print("Histogram: ", data_set.hist())
