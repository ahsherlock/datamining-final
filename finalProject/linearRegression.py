import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
plt.rcParams['figure.figsize'] = (10.0, 7.0)
# found the data set at https://www.kaggle.com/sidtwr/videogames-sales-dataset?select=Video_Games_Sales_as_at_22_Dec_2016.csv

def splitThatShit(dataframe):
    dataCopy = dataframe.copy()
    trainingData = dataCopy.sample(frac=0.75, random_state=0)
    testingData = dataCopy.drop(trainingData.index)
    return trainingData, testingData

# plots Global sales against another column of data to show correlations
def showDataCorrelation(columnName, dataFrame):
    Y = dataFrame["Global_Sales"]
    X = dataFrame[columnName]
    plt.title("Global_Sales vs " + columnName)
    plt.xlabel(columnName)
    plt.ylabel("Global Sales")
    plt.scatter(X, Y)
    plt.show()

def trainGradientDescent(trainX, trainY,learningRate, iterations):
    m = 0
    c = 0
    L = learningRate  # The learning Rate
    epochs = iterations  # The number of iterations to perform gradient descent
    n = float(len(trainX))  # Number of elements in X
    # Performing Gradient Descent
    for i in range(epochs):
        Y_pred = m * trainX + c  # The current predicted value of Y
        D_m = (-2 / n) * sum(trainX * (trainY - Y_pred))  # Derivative wrt m
        D_c = (-2 / n) * sum(trainY - Y_pred)  # Derivative wrt c
        m = m - L * D_m  # Update m
        c = c - L * D_c  # Update c
        return m, c


# PREPROCESSING OF THE DATA

data = pd.read_csv('vgsales2019short.csv', usecols=['Global_Sales', 'User_Score', 'Critic_Score'
    , 'Year'])
dataClean = data.dropna(subset=['User_Score', 'Critic_Score', 'Year', 'Global_Sales'])
dataClean = dataClean[dataClean['User_Score'] != 'tbd']
dataClean['User_Score_Numeric'] = pd.to_numeric(dataClean['User_Score'])
dataClean['Critic_Score_Numeric'] = pd.to_numeric(dataClean['Critic_Score'])
dataClean['User_Score_Numeric'] = dataClean['User_Score_Numeric'].apply(lambda x: 10 * x)
print("ORIGINAL DATA BEFORE THE SPLIT")
print(dataClean.head())
print(dataClean.describe())
print("\n**************************************************")
print("\n")
print("\n")
print("\n")

training, testing = splitThatShit(dataClean)
print("Training Data")
print(training.describe())
print(training.head(25))
print(training['Global_Sales'])
print("\n")
print("Testing Data")
print(testing.describe())
print(testing.head(25))
print("\n")
print("Original Data")
print(dataClean.describe())
print(dataClean.head(25))

# Show data correlations
showDataCorrelation("User_Score_Numeric", dataClean)
showDataCorrelation("Critic_Score_Numeric", dataClean)
# Y = dataClean.iloc[:, 1]
# X = dataClean.iloc[:, 2]
# plt.xlabel("User Score")
# plt.ylabel("Global Sales")
# plt.scatter(X, Y)
# plt.show()
#
# X = dataClean.iloc[:, 4]
# plt.xlabel("Critic Score")
# plt.ylabel("Global Sales")
# plt.scatter(X, Y)
# plt.show()
#
# X = dataClean.iloc[:, 0]
# plt.xlabel("Year of Release")
# plt.ylabel("Global Sales")
# plt.scatter(X, Y)
# plt.show()

# X = data.iloc[:, 0]
# Y = data.iloc[:, 1]
# plt.scatter(X, Y)
# plt.show()
#

print("BUILDING MODEL")
# Building the model for Gradient Descent
X = training.iloc[:, 4]
print(X.describe())
Y = training.iloc[:, 1]
print(Y.describe())
m, c = trainGradientDescent(X, Y, 0.0001, 1000)
testingX = testing.iloc[:, 4]

# Performing Gradient Descent
# for i in range(epochs):
#     Y_pred = m * X + c  # The current predicted value of Y
#     D_m = (-2 / n) * sum(X * (Y - Y_pred))  # Derivative wrt m
#     D_c = (-2 / n) * sum(Y - Y_pred)  # Derivative wrt c
#     m = m - L * D_m  # Update m
#     c = c - L * D_c  # Update c


print(m, c)
# Making predictions
Y_pred = m*testingX + c

plt.scatter(X, Y)
plt.xlabel("User Score")
plt.ylabel("Global Sales")
plt.title("Gradient Descent Linear Regression")
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()


# Build Model for Least Squares

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
