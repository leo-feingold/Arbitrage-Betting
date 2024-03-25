import pandas as pd
import numpy as np


df = pd.read_csv('/Users/leofeingold/Documents/GitHub/Arbitrage-Betting/fullData.csv').rename(columns = lambda x: x.lower())
df = df.assign(
    absDiff = lambda x: abs(x.difference)
)


df['expected value (%)'] = pd.to_numeric(df['expected value (%)'], errors='coerce')

minEV = 9
mask = df['expected value (%)'] >= minEV
df = df[mask]


# Function to calculate the probability of losing every bet
def probability_of_losing_every_bet(df):
    return (1 - df['probability']).prod()


# Function to calculate the probability of winning every bet
def probability_of_winning_every_bet(df):
    return (df['probability']).prod()


print(f"Probability of losing every bet (min EV = {minEV}):", probability_of_losing_every_bet(df))
print(f"Probability of winning every bet (min EV = {minEV}):", probability_of_winning_every_bet(df))
print(df.probability.describe())

