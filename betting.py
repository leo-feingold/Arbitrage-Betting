import pandas as pd
df2023 = pd.read_csv('Betting - Odds via fandual March 30th, 2023 preseason .csv')


def calcProbability(difference, threshold):
    winCounter = 0
    totalBets = 0
    for i in range(len(df2023)):
        if(
            (abs(df2023['difference'].iloc[i])) > difference - threshold and
            (abs(df2023['difference'].iloc[i])) < difference + threshold
            ):
            #print("Index: " , df2023.index[i] , " Difference: " , df2023['difference'].iloc[i], "Hit or Miss: " , df2023['hit/miss'].iloc[i])
            if (df2023['hit/miss'].iloc[i]) == "hit":
                winCounter += 1
            if (df2023['hit/miss'].iloc[i]) != "null":
                totalBets += 1

    print("Difference: " , difference)
    print("Threshold: " , threshold)
    return (winCounter/totalBets)


print("Historical Probabiltiy of Success: " , calcProbability(4,0.3))