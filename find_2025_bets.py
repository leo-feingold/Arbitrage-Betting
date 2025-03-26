# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("/Users/leofeingold/Desktop/Arbitrage-Betting/2025_season_bets.csv")
    return df

def calc_expected_value(prob_win, odds):
    if odds > 0:
        profit = odds / 100
    else:
        profit = 100 / np.abs(odds)
    
    loss = 1
    prob_loss = 1 - prob_win
    ev = (prob_win * profit) - (prob_loss * loss)
    return ev


def calc_probability_and_ev_of_winning(df):
    books = []
    for book in books:
        df[f"prob_win_{book}"] = -0.302 * np.exp(-0.131 * df[f"diff_{book}"]) + 0.773

        # basically if the books_projection is over the fangraphs projection, we would take the under. 
        if df[f"{book}_line"] > df["fangraphs_line"]:
            df[f"expected_value_{book}"] = calc_expected_value(df[f"prob_win_{book}"], df[f"{book}_odds_under"])

        else:
            df[f"expected_value_{book}"] = calc_expected_value(df[f"prob_win_{book}"], df[f"{book}_odds_over"])

