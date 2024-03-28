import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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
    #df = df[df['Value'] <= 8] #interesting, but this changes sample size
    return df.dropna()


def plotDataNonLinear():
    sample_size = countSampleSize()
    df = formData()
    plt.scatter(df['Value'], df['Probability'], marker='o', color='blue', label='Data')

    # Define a nonlinear function to fit the data
    def nonlinear_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Fit the nonlinear function to the data
    popt, pcov = curve_fit(nonlinear_func, df['Value'], df['Probability'])

    # Plot the fitted curve
    x_values = np.linspace(min(df['Value']), max(df['Value']), 100)
    plt.plot(x_values, nonlinear_func(x_values, *popt), color='red', label='Nonlinear Best Fit')

    plt.xlabel('Difference')
    plt.ylabel('Probability')
    plt.title('Probability vs Difference: Beating Vegas')
    plt.legend()
    plt.grid(True)
    equation = f'y = {popt[0]:.3f} * exp(-{popt[1]:.3f} * x) + {popt[2]:.3f}'
    plt.text(0.3, 0.15, equation, fontsize=12, color='black', transform=plt.gca().transAxes)
    plt.suptitle(f"Sample Size: {sample_size} Bets")
    #plt.show()
    return popt


def calcPotentialPayout(odds):
    # assuming 100 dollar bet
    if odds >= 100:
        return odds
    else:
        return 100/abs(odds) * 100


def calcEV(odds, book, team, bet, csv_file, line, proj):
    coef = plotDataNonLinear()
    difference = (line - proj)
    probability = coef[0] * np.exp(-1 * coef[1] * abs(difference)) + coef[2]
    winAmount = calcPotentialPayout(odds)
    expected_value = winAmount*probability - 100*(1 - probability)
    print(f"Probability: {probability}")
    print(f"Expected ROI at {odds} and {difference} difference: {expected_value}%")


    # Create DataFrame to write to CSV
    data = {'Book': [book],
            'Odds': [odds],
            'Bet Type': [bet],
            'Team': [team],
            'Expected Value (%)': [expected_value],
            'Probability': [probability],
            'Difference': [difference],
            'Line': [line],
            'Projection': [proj]}

    df = pd.DataFrame(data)
    
    # Append data to CSV file (this is chatGPT not sure what it does)
    with open(csv_file, 'a') as f:
      df.to_csv(f, header=f.tell()==0, index=False)


    return expected_value


def main():
    odds = -120
    book = "Fanduel" # if same defualt to Fanduel
    team = ""
    bet = "Under"
    csv_file = "adjustedData.csv"  # path to the CSV file
    line = 81.5
    proj = 79
    calcEV(odds, book, team, bet, csv_file, line, proj)


if __name__ == "__main__":
    main()
