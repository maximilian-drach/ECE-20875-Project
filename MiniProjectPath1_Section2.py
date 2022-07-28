import pandas as pd
import matplotlib.pyplot as plt

dfTempVsTraffic = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv', usecols = ['High Temp', 'Low Temp', 'Total'])
tempVals = dfTempVsTraffic[['High Temp', 'Low Temp']]
dfTempVsTraffic ['Average Temp'] = tempVals.mean(numeric_only = True, axis = 1)

print (dfTempVsTraffic)

x = dfTempVsTraffic['Average Temp']
print("x: ", x)
y = dfTempVsTraffic['Total']
print("y: ", y)
plt.scatter(x,y)
plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
#plt.yticks([0, 5000, 10000, 15000, 20000, 25000, 30000])
plt.xlabel('Average Temperature')
plt.ylabel('Total Bike Traffic')
plt.title('Average Temperature vs Total Bike Traffic')
plt.show()