import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import seaborn as sns
from scipy.stats import norm
from os.path import dirname, abspath

if __name__ == '__main__':
    root_dir = abspath(dirname(dirname(__file__)))
    fci = pd.read_csv(f'{root_dir}/riskindex/fci.csv')
    fci['date'] = pd.to_datetime(fci.iloc[:, 0])
    fci.index = fci['date']
    fci.drop('date', axis=1, inplace=True)
    fci.fillna(method='bfill', inplace=True)
    fci.dropna(inplace=True)
    index = '.TED G Index'
    y = fci[index]
    fig, ax = plt.subplots(3, 2, constrained_layout=True)
    fci[index].plot(xlabel='Date', ylabel=index, title=[index], subplots=True, ax=ax[0, 0])
    fci[index].diff().plot(xlabel='Date', ylabel='diff', title=['Diff'], subplots=True, ax=ax[0, 1])
    sns.histplot(data=y, ax=ax[1, 0], kde=True, stat='density').set(title='Histogram of ' + index)
    mu = np.mean(y)
    std = np.std(y)
    x = np.linspace(min(y), max(y), 100)
    p = norm.pdf(x, mu, std)
    ax[1, 0].plot(x, p, 'k', linewidth=2)
    plot_acf(x=y, title='Autocorrelation', ax=ax[1, 1])
    ax[1, 1].set_xlabel('Lags')
    ax[1, 1].set_ylabel('ACF value')
    plot_pacf(x=y, title='Partial Autocorrelation', ax=ax[2, 0])
    ax[2, 0].set_xlabel('Lags')
    ax[2, 0].set_ylabel('PACF value')
    plt.show()


