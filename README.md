# Final Project
## Due: 7/29/2022 11:59pm ET

Put yourself in the shoes of a data scientist being given a data set and asked to draw conclusions from it. Your job will be to understand what the data is showing you, design the analyses you need, justify those choices, draw conclusions from running the analyses, and explain why they do (or do not) make sense.

We are deliberately not giving you detailed directions on how to solve these problems, but feel free to come to office hours to brainstorm.

## Objectives

There are two possible paths through this project:

1. You may use data set #1, which captures information about bike usage in New York City. See below for the analysis questions we want you to answer.
2. You may use data set #2, which captures information about stock history for Apple and Google company. See below for the analysis questions we want you to answer.

## Partners

You will be working in a learning community. 
After your first meeting you must fill out this form (https://forms.gle/UNFDViSzRe8hcRUu7). 

## Path 1: Bike traffic

The `NYC_Bicycle_Counts_2016_Corrected.csv` gives information on bike traffic across a number of bridges in New York City. In this path, the analysis questions we would like you to answer are as follows:

1. You want to install sensors on the bridges to estimate overall traffic across all the bridges. But you only have enough budget to install sensors on three of the four bridges. Which bridges should you install the sensors on to get the best prediction of overall traffic?
2. The city administration is cracking down on helmet laws, and wants to deploy police officers on days with high traffic to hand out citations. Can they use the next day's weather forecast(low/high temperature and precipitation) to predict the total number of bicyclists that day? 
3. Can you use this data to predict what *day*(Monday to Sunday) is today based on the number of bicyclists on the bridges?
   
### Dataset description
>**High Temperature**: The highest temperature in one day in ˚F. \
>**Low Temperature**: The lowest temperature in one day in ˚F.\
>**Precipitation**: rain drop height (in inch).\
>**Total**: The total for bike usage on four bridges in each day.

## Path 2: Stock history

The `APPL.csv` and `GOOG.csv` contains the stock information for Apple and Google. For each dataset, each column corresponds to each day's stock price attributes. In this path, the analysis questions we would like you to answer are as follows:

1. Can you predict the next day closing price for each stock (`close`)? How good is your prediction? What about predicting for the next week/month?
2. Can you identify any long term or short term patterns in the data that correspond to historic economic shifts in the stock market (e.g., 2008 recession)? Consider using clustering, grouping, regression, or any other analytical method to complete this task. Please describe why you choose such method(s) in your analysis and explain your results.
3. If you want to invest only one company, which one would you choose? Why?

### Dataset description

>**Date**: Date time for the stock information.\
>**Open, High, Low, Close, Volume**: Five important statistics in stock price. More details can be found [here](https://analyzingalpha.com/open-high-low-close-stocks). \
>**Adjusted close**: Price adjusted for splits and dividend and/or capital gain distributions.

### Hint
1. Split the dataset into training and test dataset first, and then apply normalization to training dataset. Make sure to apply the same mean and standard deviation (obtained from normalization of the training dataset) to test dataset when evaluating your model.
2. The predicted value for each day/week/month can be created by shifting the `close` attribute down in the table by a day/week/month.

## What to turn in
For each draft (including the final project) you should push to your team's leader Github repository the `.py` code and pdf report. Keep the pdf Drafts in the same repository.  

#### Draft 0 
* `Draft0.pdf`
Until Section 2

#### Draft 1
* `Draft1.pdf`
Until Section 3

#### Final report 

* `report.pdf`: A project report, which should consist of:

  * Section 1 (Group infomration):  with the names of the team members (maximum of two), your Purdue username(s), and the path (1 or 2) you have taken. 
  * Section 2 (Description of the dataset) :  describing the dataset you are working with, all variables need to be included. In addition, you should include at least     one visual that help descriobe one or multiple variables. The description of the visual should be included too. 
  * Section 3 (Methods): Describing the analyses you chose to use for each analysis question (with a paragraph or two justifying why you chose that analysis and what       you expect the analysis to tell you). This section should not include results. For example if you plan to include in your analysis a linear regression model and       you will use residual plots to evaluate the fit. You must explain what tool you are going to use and why but not the results from applying the tool. 
  * Section 4 (Results): describing the results of each analysis, and what your answers to the questions are based on your results. Visual aids are helpful here, you       are expected to at least have one visual aid that is helpful to back up your conclusions. 
    Note that it is OK if you do not get "positive" answers from your analysis, but you must explain why that might be.


* For the final report the repository must contain: All Python `.py` code files you wrote to complete the analysis steps. Draft0.pdf, Draft1.pdf, report.pdf 
