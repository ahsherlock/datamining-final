import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

#url = 'https://raw.githubusercontent.com/danieljaouen/DS-Unit-1-Sprint-1-Dealing-With-Data/master/module1' \
#      '-afirstlookatdata/Video_Games_Sales_as_at_22_Dec_2016.csv '
#data = pd.read_csv(url, usecols=["Global_Sales", "Critic_Score", "User_Score"])
#print(data)


# Preprocessing Input data
data = pd.read_csv('dumbData.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
#plt.scatter(X, Y)
#plt.show()

# Building the model
m = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X))  # Number of elements in X

# Performing Gradient Descent
for i in range(epochs):
    Y_pred = m * X + c  # The current predicted value of Y
    D_m = (-2 / n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2 / n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c

print(m, c)
# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y)
plt.xlabel("population in 10,000s")
plt.ylabel("profit in 10,000$")
plt.title("FoodTruck Profit")
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()