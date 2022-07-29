import pandas as pd
import matplotlib.pyplot as plt

dfTempVsTraffic = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv', usecols = ['High Temp', 'Low Temp', 'Total'])
tempVals = dfTempVsTraffic[['High Temp', 'Low Temp']]
#print (dfTempVsTraffic.info())
dfTempVsTraffic ['Average Temp'] = tempVals.mean(axis = 1) #uses high and low temp columns to calculate average temp column

dfTempVsTraffic['Total'] = dfTempVsTraffic['Total'].str.replace(',','') #removes commas from total column
dfTempVsTraffic['Total'] = dfTempVsTraffic ['Total'].astype(float, errors = 'raise') #converts total column from object type to float
#print (dfTempVsTraffic.info())

#print (dfTempVsTraffic)

#creating and plotting the graph
x = dfTempVsTraffic['Average Temp']
y = dfTempVsTraffic['Total']
plt.scatter(x,y)
plt.xticks([30, 40, 50, 60, 70, 80, 90, 100])
plt.yticks([0, 5000, 10000, 15000, 20000, 25000, 30000])
plt.grid()
plt.xlabel('Average Temperature ($^\circ$F)')
plt.ylabel('Total Bike Traffic')
plt.title('Average Temperature Vs. Total Bike Traffic')
plt.show()
