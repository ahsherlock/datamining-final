import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import requests
plt.rcParams['figure.figsize'] = (12.0, 9.0)
#url = 'https://raw.githubusercontent.com/danieljaouen/DS-Unit-1-Sprint-1-Dealing-With-Data/master/module1' \
      #'-afirstlookatdata/Video_Games_Sales_as_at_22_Dec_2016.csv '


#data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv', usecols=["Global_Sales", "Critic_Score", "User_Score"])
#print(data.head())
#video_game_data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv', delimiter=',')
#print(video_game_data.head())
#video_game_data2 = video_game_data.dropna(subset=['User_Score', 'Critic_Score', 'Year_of_Release'])
#video_game_data2 = video_game_data2[video_game_data2['User_Score'] != 'tbd']
#video_game_data2['User_Score_Numeric'] = pd.to_numeric(video_game_data2['User_Score'])
#video_game_data2['Critic_Score_Numeric'] = pd.to_numeric(video_game_data2['Critic_Score'])
#video_game_data2['User_Score_Numeric'] = video_game_data2['User_Score_Numeric'].apply(lambda x:10*x)
#print(video_game_data2['Global_Sales'].head())
#print(video_game_data2['User_Score_Numeric'])
#video_game_data2.plot.scatter('User_Score_Numeric', 'Global_Sales')
#plt.scatter(video_game_data2['User_Score_Numeric'], video_game_data2['Global_Sales'])
#plt.show()
#X = data.iloc[:, 1]
#Y = data.iloc[:, 0]
#plt.xlabel("Critic Score")
#plt.ylabel("Global Sales")
#plt.title("Global Sales VS Critic Scores")
#plt.scatter(X,Y)
#plt.show()
#X = data.iloc[:, 2]
#Y = data.iloc[:, 0]
#plt.xlabel("User Score")
#plt.ylabel("Global Sales")
#plt.title("Global Sales VS User Scores")
#plt.plot(X,Y)
#plt.show()


