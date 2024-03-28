# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


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
def addCols(df, betAmount):
    df = df.assign(
        profit = lambda x: calcPotentialPayout(x["odds"], betAmount),
        loss = lambda x: -1 * betAmount,
        betAmount = betAmount
    )
    return df

# create the dataframe with those columns
moneyPerBet = 100
df = addCols(df, moneyPerBet)


# here is the random sample and simulation:
def randomSample(df):
    totalWins = 0
    totalLosses = 0
    distribution = []
    totalMoney = 0

    for sample in range(1000): # 1000 seasons of simulations 
        winCount = 0
        loseCount = 0
        money = 0
        for i in range(len(df)):
            random_float = np.random.random()  
            if df.probability.iloc[i] >= random_float: # it is a win if the random float is less than or equal to the probability
                winCount += 1
                money += df.profit.iloc[i]
            else:
                loseCount += 1
                money += df.loss.iloc[i]

        distribution.append(winCount)

        print(f"Num Wins: {winCount}, Num Losses: {loseCount}, Result: {winCount/(winCount + loseCount)}, Money: {money}")
        totalMoney += money
        totalWins += winCount
        totalLosses += loseCount

    win_counts = Counter(distribution)
    result = totalWins / (totalWins + totalLosses)
    resultStr = f"Num Wins: {totalWins}, Num Losses: {totalLosses}, Result: {result}, Average Money (per year): {totalMoney/1000}"

    prob_distribution = {k: v / 1000 for k, v in win_counts.items()}

    # Plot the probability distribution
    plt.bar(prob_distribution.keys(), prob_distribution.values())
    plt.xlabel('Winning Bets (per season)')
    plt.ylabel('Probability of Occurance')
    plt.title('Probability Distribution of Win Counts')
    plt.suptitle(f"Dollars Per Bet: {moneyPerBet}, Simulated Seasons: 1000, Average Profit: ${totalMoney/1000:.2f}, Min ROI: {minEV}%, Number of Bets: {len(df)}")
    plt.show()

    return resultStr

print(randomSample(df))