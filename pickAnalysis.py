# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the data and create a new column called absDiff that is the absolute value of the difference column 
df = pd.read_csv('/Users/leofeingold/Documents/GitHub/Arbitrage-Betting/fullData.csv').rename(columns = lambda x: x.lower())
df = df.assign(
    absDiff = lambda x: abs(x.difference)
)

# convert the ROI column into numeric values
df['expected value (%)'] = pd.to_numeric(df['expected value (%)'], errors='coerce')

# Restrict the bets down to only bets with at least an 8% ROI (8 is somewhat arbitralily chosen...)
minEV = 8
mask = df['expected value (%)'] >= minEV
df = df[mask]

# payout 
def calcPotentialPayout(odds, betAmount):
    return np.where(odds >= 100, (odds * betAmount)/100, betAmount / np.abs(odds) * 100)

# add the profit, loss and betAmount columns to the data frame
def probDist(df, betAmount):
    df = df.assign(
        profit = lambda x: calcPotentialPayout(x["odds"], betAmount),
        loss = lambda x: -1 * betAmount,
        betAmount = betAmount
    )
    return df

# create the dataframe with those columns
df = probDist(df, 200)


# here is the random sample and simulation:
def randomSample(df):
    totalWins = 0
    totalLosses = 0
    distribution = []

    for samples in range(1000): # 1000 seasons of simulations 
        winCount = 0
        loseCount = 0
        for i in range(len(df)):
            random_float = np.random.random()  
            if df.probability.iloc[i] >= random_float: # it is a win if the random float is less than or equal to the probability
                winCount += 1
            else:
                loseCount += 1

        #for j in range(len(df)):
            #if winCount == i:
                #want to create a binomial distribution of the results         

        print(f"Num Wins: {winCount}, Num Losses: {loseCount}, Result: {winCount/(winCount + loseCount)}")
        totalWins += winCount
        totalLosses += loseCount

    result = totalWins / (totalWins + totalLosses)
    resultStr = f"Num Wins: {totalWins}, Num Losses: {totalLosses}, Result: {result}"
    return resultStr

print(randomSample(df))

'''

sample result from function:

Num Wins: 12, Num Losses: 2, Result: 0.8571428571428571
Num Wins: 13, Num Losses: 1, Result: 0.9285714285714286
Num Wins: 8, Num Losses: 6, Result: 0.5714285714285714
Num Wins: 11, Num Losses: 3, Result: 0.7857142857142857
Num Wins: 10, Num Losses: 4, Result: 0.7142857142857143
Num Wins: 7, Num Losses: 7, Result: 0.5
Num Wins: 8, Num Losses: 6, Result: 0.5714285714285714
Num Wins: 9, Num Losses: 5, Result: 0.6428571428571429
Num Wins: 12, Num Losses: 2, Result: 0.8571428571428571
Num Wins: 7, Num Losses: 7, Result: 0.5
Num Wins: 9, Num Losses: 5, Result: 0.6428571428571429
Num Wins: 13, Num Losses: 1, Result: 0.9285714285714286
Num Wins: 6, Num Losses: 8, Result: 0.42857142857142855
Num Wins: 8, Num Losses: 6, Result: 0.5714285714285714
Num Wins: 9, Num Losses: 5, Result: 0.6428571428571429
Num Wins: 9, Num Losses: 5, Result: 0.6428571428571429
Num Wins: 12, Num Losses: 2, Result: 0.8571428571428571
Num Wins: 10, Num Losses: 4, Result: 0.7142857142857143
Num Wins: 13, Num Losses: 1, Result: 0.9285714285714286
Num Wins: 9, Num Losses: 5, Result: 0.6428571428571429
Num Wins: 8, Num Losses: 6, Result: 0.5714285714285714
Num Wins: 8, Num Losses: 6, Result: 0.5714285714285714
Num Wins: 9, Num Losses: 5, Result: 0.6428571428571429
Num Wins: 11, Num Losses: 3, Result: 0.7857142857142857
Num Wins: 10, Num Losses: 4, Result: 0.7142857142857143

'''