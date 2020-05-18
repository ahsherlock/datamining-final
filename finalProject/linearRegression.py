import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
plt.rcParams['figure.figsize'] = (10.0, 7.0)
plt.style.use('ggplot')
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

def trainLeastSquare(trainX, trainY):
    meanX = np.mean(trainX)
    meanY = np.mean(trainY)
    num = 0
    den = 0
    for i in range(len(trainX)):
        num += (trainX[i] - meanX) * (trainY[i] - meanY)
        den += (trainX[i] - meanX)**2
    coEffM = num/den
    coEffC = meanY - m*meanX
    return coEffM, coEffC


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

def predictYValues(testingX,predictedY,m,c):
    for i in range(len(testingX)):
        predictedY.append(m * testingX[i] + c)
    return predictedY

def getRSquare(X,Y,m, c):
    ss_t = 0
    ss_r = 0
    for i in range(len(X)):
        yPrediction = c + m*X[i]
        ss_t += (Y[i] - np.mean(Y))**2
        ss_r += (Y[i] - yPrediction)**2
    rSquare = 1 - (ss_r/ss_t)
    return rSquare

def getRMSE(X,Y, m , c):
    mse = 0
    for i in range(len(X)):
        yPrediction = c + m*X[i]
        mse += (Y[i] - yPrediction)**2
    rmse = np.sqrt(mse/len(X))
    return rmse

def minMaxNormalization(dataFrame):
    dataFrame = (dataFrame - dataFrame.min())/(dataFrame.max() - dataFrame.min())
    return dataFrame
def makeNumeric(dataFrame):
    dataFrame = dataFrame.astype('float')
    return dataFrame


# PREPROCESSING OF THE DATA

data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv', usecols=['Global_Sales', 'User_Score', 'Critic_Score'])
dataClean = data.dropna(subset=['User_Score', 'Critic_Score','Global_Sales'])
dataClean = dataClean[dataClean['User_Score'] != 'tbd']
dataClean = makeNumeric(dataClean)
dataClean['User_Score_Numeric'] = pd.to_numeric(dataClean['User_Score'])
dataClean['Critic_Score_Numeric'] = pd.to_numeric(dataClean['Critic_Score'])
dataClean['User_Score_Numeric'] = dataClean['User_Score_Numeric'].apply(lambda x: 10 * x)

print("ORIGINAL DATA BEFORE THE SPLIT")
print(dataClean.info())
print(dataClean.head(5))
print(dataClean.describe())
print("\n**************************************************")
print("\n")
print("\n")
print("\n")

training, testing = splitThatShit(dataClean)
print("Training Data")
print(training.info())
print(training.describe())
print(training.head(25))
#print(training['Global_Sales'])
print("\n")
print("Testing Data")
print(testing.describe())
print(testing.head(25))
print("\n")
print("Original Data")
print(dataClean.describe())
print(dataClean.head(25))

# Show data correlations
#showDataCorrelation("User_Score_Numeric", dataClean)
#showDataCorrelation("Critic_Score_Numeric", dataClean)


print("********** BUILDING MODEL USING GRADIENT DESCENT FOR USER SCORE *************")
# Building the model for Gradient Descent
X = training["User_Score_Numeric"].values
Y = training["Global_Sales"].values
Y = minMaxNormalization(Y)
m, c = trainGradientDescent(X, Y, 0.0001, 1000)
testingX = testing["User_Score_Numeric"].values
predictedY = []
predictedY = predictYValues(testingX,predictedY,m,c)
plt.plot(testingX, predictedY, c="#52b920", label="Regression Line")
plt.scatter(X, Y, c="#ef4423", label = "Scatter plot")
plt.title("Global Sales v User Score (Gradient Descent)")
plt.xlabel('USER SCORE')
plt.ylabel('GLOBAL SALES')
plt.legend()
plt.show()
print("CoEff1 = "+str(m))
print("CoEff2 = "+str(c))
#R Square
rSquareGradDecUser = getRSquare(X,Y,m,c)
print("R2 Value for gradient descent on User Score = " + str(rSquareGradDecUser))
#Root Mean Square Error
rootMeanSquareError = getRMSE(X,Y,m,c)
#mse = np.sum((predictedY - Y)**2)
#rootMeanSquareError = np.sqrt(mse/len(X))
print("The Root Mean Square Error for gradient descent on User Score = " + str(rootMeanSquareError)+ "\n\n")


print("********** BUILDING MODEL USING GRADIENT DESCENT FOR CRITIC SCORE *************")
# Building the model for Gradient Descent
X = training["Critic_Score_Numeric"].values
Y = training["Global_Sales"].values
Y = minMaxNormalization(Y)
m, c = trainGradientDescent(X, Y, 0.0001, 1000)
testingX = testing["Critic_Score_Numeric"].values
#Predict values
predictedY = m*testingX+c
plt.plot(testingX, predictedY, c="#52b920", label="Regression Line")
plt.scatter(X, Y, c="#ef4423", label="Scatter plot")
plt.title('GLOBAL SALES v. CRITIC SCORE(Gradient Descent)')
plt.xlabel('CRITIC SCORE')
plt.ylabel('GLOBAL SALES')
plt.legend()
plt.show()
print("CoEff1(m) = "+str(m))
print("CoEff2(c) = "+str(c))
#R Square
rSquareGradDecUser = getRSquare(X, Y, m, c)
print("R2 Value for gradient descent on Critic Score = " + str(rSquareGradDecUser))
#Root Mean Square Error
rootMeanSquareError = getRMSE(X, Y, m, c)
print("The Root Mean Square Error for gradient descent on Critic Score = " + str(rootMeanSquareError) + "\n\n")




print("********** BUILDING MODEL USING LEAST SQUARE FOR USER SCORE *************")
X = training['User_Score_Numeric'].values
Y = training['Global_Sales'].values
Y = minMaxNormalization(Y)
m, c = trainLeastSquare(X,Y)
testingX = testing['User_Score_Numeric'].values
Y_pred = m*X + c
print("CoEff1(m): " + str(m))
print("CoEff2(c): " + str(c))
rSquareLeastUser = getRSquare(X, Y, m, c)
print("R Square for Least Square Method on Global Sales v User Score: " + str(rSquareLeastUser))
rootMeanSquareError = getRMSE(X, Y, m, c)
print("The Root Mean Square Error: " + str(rootMeanSquareError))
print("\n\n")
plt.title("Global Sales v User Score (Least Square)")
plt.xlabel('Critic SCORE')
plt.ylabel('GLOBAL SALES')
plt.legend()
plt.scatter(X, Y, color='purple', label=('Scatter Plot'))
#predicted graph
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='green', label='Regression Line')
plt.show()



print("********** BUILDING MODEL USING LEAST SQUARE FOR CRITIC SCORE *************")
X = training['Critic_Score_Numeric'].values
Y = training['Global_Sales'].values
Y = minMaxNormalization(Y)
m, c = trainLeastSquare(X,Y)
testingX = testing['Critic_Score_Numeric'].values
Y_pred = m*testingX + c
print("CoEff1(m): " + str(m))
print("CoEff2(c): " + str(c))
rSquareLeastUser = getRSquare(X, Y, m, c)
print("R Square for Least Square Method on Global Sales v Critic Score: " + str(rSquareLeastUser))
rootMeanSquareError = getRMSE(X, Y, m, c)
print("The Root Mean Square Error: " + str(rootMeanSquareError))
print("\n\n")
plt.title("Global Sales v Critic Score (Least Square)")
plt.xlabel('USER SCORE')
plt.ylabel('GLOBAL SALES')
plt.legend()
plt.scatter(X, Y, color ='purple', label="Scatter Plot")
#predicted graph
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='green', label="Regression Line")
plt.show()

