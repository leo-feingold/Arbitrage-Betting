import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadData():

    df2023 = pd.read_csv('Betting - Odds via fandual March 30th, 2023 preseason  (1).csv')
    df2022 = pd.read_csv('Betting - odds via draftkings april 4, 2022 preseason.csv')
    df2021 = pd.read_csv('Betting - odds via draftkings feb 18, 2021 preseason (1).csv')
    df2019 = pd.read_csv('Betting - odds via Westgate Las Vegas Superbook, Feb 17, 19.csv')
    df2018 = pd.read_csv('Betting - Odds via Bovada, Mar 8, 2018 preseason (1).csv')
    df2017 = pd.read_csv('Betting - Bovada, Mar 30, 2017 preseason (1).csv')
    df2016 = pd.read_csv('Betting - odds via bookmaker, apr 2 2016 preseason.csv')

    dataFrames = [df2023, df2022, df2021, df2019, df2018, df2017, df2016]
    return dataFrames


def countSampleSize():

    dataFrames = loadData()
    voidedBets = 0

    for yearIterator in range(len(dataFrames)):
        for rowIterator in range(30):
            if((dataFrames[yearIterator]['hit/miss'].iloc[rowIterator])) == "voids":
                voidedBets += 1

    return ((len(dataFrames)*30) - voidedBets)


def formData():

    dataFrames = loadData()
    myArr = []

    for yearIterator in range(len(dataFrames)):
        for rowIterator in range(30):
            if ((dataFrames[yearIterator]['hit/miss'].iloc[rowIterator])) == "hit":
                val = 1
                currDifference = abs(dataFrames[yearIterator]['difference'].iloc[rowIterator])
                myArr.append([currDifference, val])
            elif (dataFrames[yearIterator]['hit/miss'].iloc[rowIterator]) == "miss":
                val = 0
                currDifference = abs(dataFrames[yearIterator]['difference'].iloc[rowIterator])
                myArr.append([currDifference, val])
            
    df = pd.DataFrame(myArr, columns=['Value', 'Probability'])
    df = df[df['Value'] <= 8] #interesting
    return df.dropna()


def plotDataLinear():
    sample_size = countSampleSize()
    df = formData()
    plt.scatter(df['Value'], df['Probability'], marker='o', color='blue', label='Data')
    m, b = np.polyfit(df['Value'], df['Probability'], 1)
    plt.plot(df['Value'], m*df['Value'] + b, color='red', label='Line Best Fit')
    plt.xlabel('Difference')
    plt.ylabel('Probability')
    plt.title('Probability vs Difference: Beating Vegas')
    plt.legend()
    plt.grid(True)
    equation = f'y = {m:.2f}x + {b:.2f}'
    plt.text(0.3, 0.15, equation, fontsize=12, color='black', transform=plt.gca().transAxes)
    plt.suptitle(f"Sample Size: {sample_size} Bets")
    plt.show()


def main():
    plotDataLinear()


if __name__ == "__main__":
    main()
