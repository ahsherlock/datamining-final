import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (10.0, 7.0)



# Preprocessing Input data
data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv', usecols=['Global_Sales', 'User_Score', 'Critic_Score'
    , 'Year_of_Release'])
dataClean = data.dropna(subset=['User_Score', 'Critic_Score', 'Year_of_Release'])
dataClean = dataClean[dataClean['User_Score'] != 'tbd']
dataClean['User_Score_Numeric'] = pd.to_numeric(dataClean['User_Score'])
dataClean['Critic_Score_Numeric'] = pd.to_numeric(dataClean['Critic_Score'])
dataClean['User_Score_Numeric'] = dataClean['User_Score_Numeric'].apply(lambda x: 10*x)
print(dataClean.head())
print(dataClean.describe())
# Show data correlations
Y = dataClean.iloc[:,1]
X = dataClean.iloc[:,2]
plt.xlabel("User Score")
plt.ylabel("Global Sales")
plt.scatter(X,Y)
plt.show()

X = dataClean.iloc[:,4]
plt.xlabel("Critic Score")
plt.ylabel("Global Sales")
plt.scatter(X,Y)
plt.show()

X = dataClean.iloc[:,0]
plt.xlabel("Year of Release")
plt.ylabel("Global Sales")
plt.scatter(X,Y)
plt.show()



#X = data.iloc[:, 0]
#Y = data.iloc[:, 1]
#plt.scatter(X, Y)
#plt.show()

# Building the model for Gradient Descent
# m = 0
# c = 0
#
# L = 0.0001  # The learning Rate
# epochs = 1000  # The number of iterations to perform gradient descent
#
# n = float(len(X))  # Number of elements in X
#
# # Performing Gradient Descent
# for i in range(epochs):
#     Y_pred = m * X + c  # The current predicted value of Y
#     D_m = (-2 / n) * sum(X * (Y - Y_pred))  # Derivative wrt m
#     D_c = (-2 / n) * sum(Y - Y_pred)  # Derivative wrt c
#     m = m - L * D_m  # Update m
#     c = c - L * D_c  # Update c
#
# print(m, c)
# # Making predictions
# Y_pred = m*X + c
#
# plt.scatter(X, Y)
# plt.xlabel("raddaradda")
# plt.ylabel("raddagradgsdfs")
# plt.title("Gradient Descent Linear Regression")
# plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
# plt.show()



#Build Model for Least Squares

# X_mean = np.mean(X)
# Y_mean = np.mean(Y)
#
# num = 0
# den = 0
# for i in range(len(X)):
#     num += (X[i] - X_mean)*(Y[i] - Y_mean)
#     den += (X[i] - X_mean)**2
# m = num / den
# c = Y_mean - m*X_mean
#
# print (m, c)



# Making predictions Least Squares
# Y_pred = m*X + c
# plt.xlabel("raddaradda")
# plt.ylabel("raddagradgsdfs")
# plt.title("Least Squares Linear Regression")
# plt.scatter(X, Y) # actual
# plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red') # predicted
# plt.show()

