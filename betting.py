import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

threshold = 0.3
minTotalBets = 3


def loadData():

    df2023 = pd.read_csv('Betting - Odds via fandual March 30th, 2023 preseason  (1).csv')
    df2022 = pd.read_csv('Betting - odds via draftkings april 4, 2022 preseason.csv')
    df2021 = pd.read_csv('Betting - April 1, 2021, BetMGM, preseason.csv')
    df2019 = pd.read_csv('Betting - March 28, 2019 preseason.csv')
    df2018 = pd.read_csv('Betting - March 29, 2018 preseason.csv')
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


def calcProbability(difference, threshold):

    dataFrames = loadData()
    
    winCounter = 0
    totalBets = 0

    for yearIterator in range(len(dataFrames)):
        for rowIterator in range(30):
            if(
                (abs(dataFrames[yearIterator]['difference'].iloc[rowIterator])) >= (difference - threshold) and
                (abs(dataFrames[yearIterator]['difference'].iloc[rowIterator])) <= (difference + threshold)
                ):
                #print("Index: " , df2023.index[rowIterator] , " Difference: " , df2023['difference'].iloc[rowIterator], "Hit or Miss: " , df2023['hit/miss'].iloc[rowIterator])
                #may still need to add some data processing here
                if ((dataFrames[yearIterator]['hit/miss'].iloc[rowIterator])) == "hit":
                    winCounter += 1
                if ((dataFrames[yearIterator]['hit/miss'].iloc[rowIterator])) != "voids":
                    totalBets += 1
    

    
    if totalBets > minTotalBets: 
        return (winCounter/totalBets)



def createNewDataFrame():
    myArr = []
    i = 0
    while i < 6.9:
        probability = calcProbability(i, threshold)
        myArr.append([i, probability])
        i += 0.1
    df = pd.DataFrame(myArr, columns=['Value', 'Probability'])
    return df.dropna()

def plotDataLinear():
    sample_size = countSampleSize()
    df = createNewDataFrame()
    plt.scatter(df['Value'], df['Probability'], marker='o', color='blue', label='Data')
    m, b = np.polyfit(df['Value'], df['Probability'], 1)
    plt.plot(df['Value'], m*df['Value'] + b, color='red', label='Line Best Fit')
    plt.xlabel('Difference')
    plt.ylabel('Probability')
    plt.title('Probability vs Difference: Beating Vegas')
    plt.legend()
    plt.grid(True)
    equation = f'y = {m:.3f}x + {b:.3f}'
    plt.text(0.5, 0.9, equation, fontsize=12, color='black', transform=plt.gca().transAxes)
    plt.suptitle(f"Threshold: {threshold}, Min Total Bets: {minTotalBets}, Sample Size: {sample_size} Bets")
    plt.show()


def main():
    #For single data point mode:
    #print("Historical Probabiltiy of Success:" , calcProbability(5,0.5))
    plotDataLinear()



if __name__ == "__main__":
    main()
