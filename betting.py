import pandas as pd


def calcProbability(difference, threshold):
    df2023 = pd.read_csv('Betting - Odds via fandual March 30th, 2023 preseason .csv')
    df2022 = pd.read_csv('Betting - odds via draftkings april 4, 2022 preseason.csv')
    df2021 = pd.read_csv('Betting - odds via draftkings feb 18, 2021 preseason.csv')
    df2019 = pd.read_csv('Betting - odds via Westgate Las Vegas Superbook, Feb 17, 19.csv')
    df2018 = pd.read_csv('Betting - Odds via Bovada, Mar 8, 2018 preseason.csv')
    df2017 = pd.read_csv('Betting - Odds via Atlantis, Feb 10, 2017 preseason.csv')

    dataFrames = [df2023, df2022, df2021, df2019, df2018, df2017]

    winCounter = 0
    totalBets = 0

    for yearIterator in range(len(dataFrames)):
        for rowIterator in range(len(df2023)):
            if(
                (abs(dataFrames[yearIterator]['difference'].iloc[rowIterator])) >= (difference - threshold) and
                (abs(dataFrames[yearIterator]['difference'].iloc[rowIterator])) <= (difference + threshold)
                ):
                #print("Index: " , df2023.index[rowIterator] , " Difference: " , df2023['difference'].iloc[rowIterator], "Hit or Miss: " , df2023['hit/miss'].iloc[rowIterator])
                if (dataFrames[yearIterator]['hit/miss'].iloc[rowIterator]) == "hit":
                    winCounter += 1
                if (dataFrames[yearIterator]['hit/miss'].iloc[rowIterator]) != "null":
                    totalBets += 1

    print("Difference: " , difference)
    print("Threshold: " , threshold)
    print("Number of Wins :" , winCounter)
    print("Total Bets: " , totalBets)
    return (winCounter/totalBets)



def main():
    print("Historical Probabiltiy of Success: " , calcProbability(3,0.3))

if __name__ == "__main__":
    main()
