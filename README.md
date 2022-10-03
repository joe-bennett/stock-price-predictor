# Description 

The objective is to create a Vector Autoregression model that takes in last six months' worth of historical stock price data and index values and forecasts the close price for the stock for the  next 7 days. Specifically this model is initially predicting the price for Baker Hughes common stock close price not only with its own historical data, but also historical data for one of the company's biggest customers Nabors drilling, and the Dow Jones Oil and Gas index.  The stock historical data for both companies was used to create the technical analysis values and used as features as well.  The idea is that if we can predict the forward rolling week long close price with accuracy it can be used for options trading strategies.


# Goals

My goal is to create and continue to refine this model so that I may forecast Baker Hughes 7 day forward rolling stock prices so that I may use it as part of potential swing trading option strategy indefinately into the future.


# Planning

data science across all domains can usually be generalized as the following steps. I use this as a framework for making my plan.

Planning- writing out a timeline of actionable items, when the MVP will be finished and how to know when it is, formulate initial questions to ask the data.

Acquisition- Gather my data and bring all necessary data into my python enviroment from yahoo finance and save locally

Preparation- this is blended with acquisition where I will clean and tidy the data and split into my train and test 

Exploration/Pre-processing- where i will create visualizations and conduct hypothesis testing to select and engineer features that impact the target variable.

Modeling- based on what i learn in the exploration of the data I will select the useful features and feed into VAR model and evaluate performance of each various lag value selections

Delivery- create a final report that succintly summarizes what I did, why I did it and what I learned in order to make recommendations



## 1
H_null=  historical values in series (x) do not have a causal relationship (historical values in regression equation = 0) with values in series (y)

H_a= I reject the null hypothesis 

## 2 
H_null= There is a unit root present (non-stationarity) in the time series 

H_a= There is stationarity in the time series data





# Data dictionary 

column name                             description

open                                   the Baker Hughes open price 

high                                   the Baker Hughes intraday high price

low                                    the Baker Hughes intraday low price

close                                  the Baker Hughes daily close price

adj close                              the Baker Hughes close price adjusted for stock splits and dividends

volume	                               number of shares of Baker Hughes bought/sold daily

RSI14	                               14 day relative strength index score for Baker Hughes

EMA10	                               10 day exponnential moving average for Baker Hughes stock

nbrs adj close                        close price for Nabors adjusted for stock split and dividends

nbrs volume                            number of share bought/sold Nabors shares daily

nbrs rsi 14                             14 day relative strength index for Nabors stock

nbrs ema10                              10 day exponnential moving average for Nabors stock

index level                             daily Dow Jones Oil and Gas index level





# how to reproduce my work

To reproduce my work you will need to import all the CSV files locally  as well as my wrangle.py file. you will need all the libraries available to import that I have listed at the top of my workbook.  with that my report will run completely from the top down. 



# Executive Summary

My goals mentioned above to create a model to predict Baker Hughes stock price 7 days out was a success- that is, it significantly beat my baseline according to RMSE scores. My recommendation is that this model be used with the most recent historical data. I retrieved my data for this project via yahoo finance last 6 mo historical data download to CSV. Anyone can go online and do this themselves and download the last 6 month data for Baker Hughes,Nabors, and the index and it will work. My model RMSE was 3.8 and the avg price was approx 30 a share. In light of the extreme volatility the market has had and was captured in the historical data used I am satisfied that I was able to get within 3.80 a share on avg. 