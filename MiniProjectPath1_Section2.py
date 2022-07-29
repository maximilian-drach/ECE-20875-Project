import pandas as pd
import matplotlib.pyplot as plt

#VISUAL 1
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

#trendline
p = np.polyfit(x, y, 1)
trendline = np.poly1d(p)
plt.plot(x, trendline(x))

plt.show()

#VISUAL 2
dfBikePerBridge = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv', usecols = ['Brooklyn Bridge','Manhattan Bridge','Williamsburg Bridge','Queensboro Bridge','Total'])
dfBikePerBridge = dfBikePerBridge.replace(',','', regex = True) #removes commas from all columns
dfBikePerBridge = dfBikePerBridge.astype(int, errors = 'raise') #converts all columns from object type to float

#taking the mean of every column to find average bike traffic value
BrooklynAvg = int(dfBikePerBridge['Brooklyn Bridge'].mean())
ManhattanAvg = int(dfBikePerBridge['Manhattan Bridge'].mean())
WilliamsburgAvg = int(dfBikePerBridge['Williamsburg Bridge'].mean())
QueensboroAvg = int(dfBikePerBridge['Queensboro Bridge'].mean())

#creating and plotting the graph
fig, ax = plt.subplots(figsize = (7,4))
x = ['Brooklyn \nBridge','Manhattan \nBridge','Williamsburg \nBridge','Queensboro \nBridge']
y = [BrooklynAvg, ManhattanAvg, WilliamsburgAvg, QueensboroAvg]
plt.bar(x,y)
plt.xlabel('Bridge Name')
plt.ylabel('Average Bike Traffic')
plt.yticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000])
plt.title('Average Bike Traffic Per Bridge')
for index in range(len(x)):
    ax.text(x[index], y[index], y[index], size = 12) #adds data labels
plt.grid(axis = 'y')
plt.show()
