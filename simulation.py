import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alphas import *

def visualizer(data):
    data.set_index('Date', inplace=True)
    close_values = data['Close']
    alpha_values = data['alpha']
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Close Prices
    ax1.plot(close_values, label='Close Prices', color='b')
    ax1.set_ylabel('Close Price', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create a second y-axis to plot alpha values
    ax2 = ax1.twinx()
    ax2.plot(alpha_values, label='Alpha', color='g')
    ax2.set_ylabel('Alpha', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    plt.title('Close Prices and Alpha Over Time')
    ax1.set_xlabel('Date')
    ax1.grid(True)
    fig.tight_layout()  # To ensure there's no overlap with the y-axis labels
    plt.show()

def simulator(data):
    results = momentum(data)
    print(results)
    data['alpha'] = results
    visualizer(data)

data = pd.read_csv('../data/stocks/AAPL.csv', parse_dates=['Date'])
simulator(data)