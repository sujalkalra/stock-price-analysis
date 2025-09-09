import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def fetch_data(ticker, start_date, end_date):
    """Fetches historical stock data."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].to_numpy().flatten()

def calculate_moving_average(data, window_size):
    """Calculates the simple moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def calculate_daily_returns(data):
    """Calculates daily returns."""
    return np.diff(data) / data[:-1] * 100

def calculate_volatility(returns):
    """Calculates the volatility (standard deviation of returns)."""
    return np.std(returns)

def calculate_bollinger_bands(data, window_size):
    """Calculates Bollinger Bands."""
    rolling_mean = calculate_moving_average(data, window_size)
    # Calculate rolling_std correctly
    rolling_std = np.array([np.std(data[i:i+window_size]) for i in range(len(rolling_mean))])
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    return upper_band, lower_band, rolling_mean

def plot_results(prices, sma, returns, volatility, upper_band, lower_band, window_size, ticker):
    """Plots the stock analysis results."""
    plt.figure(figsize=(12, 8))

    # Plot stock price with moving average and Bollinger Bands
    plt.subplot(2, 1, 1)
    plt.plot(prices, label='Stock Price')
    plt.plot(np.arange(window_size-1, len(prices)), sma, label=f'{window_size}-day Moving Average')
    plt.plot(np.arange(window_size-1, len(prices)), upper_band, linestyle='--', color='gray', label='Upper Bollinger Band')
    plt.plot(np.arange(window_size-1, len(prices)), lower_band, linestyle='--', color='gray', label='Lower Bollinger Band')
    plt.fill_between(np.arange(window_size-1, len(prices)), lower_band, upper_band, alpha=0.2)
    plt.title(f'{ticker} Stock Price, Moving Average, and Bollinger Bands')
    plt.legend()

    # Plot daily returns and volatility
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(returns)), returns, label='Daily Returns', color='orange')
    plt.axhline(y=volatility, color='r', linestyle='--', label=f'Volatility ({volatility:.2f}%)')
    plt.title('Daily Returns and Volatility')
    plt.legend()

    plt.tight_layout()
    plt.savefig('stock_analysis.png')
    print("Stock analysis plot saved to stock_analysis.png")
    plt.show()

def main():
    """Main function to run the stock analysis."""
    # Parameters
    ticker = 'AAPL'
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    window_size = 50

    # Perform analysis
    prices = fetch_data(ticker, start_date, end_date)
    sma = calculate_moving_average(prices, window_size)
    returns = calculate_daily_returns(prices)
    volatility = calculate_volatility(returns)
    upper_band, lower_band, _ = calculate_bollinger_bands(prices, window_size)

    # Plot results
    plot_results(prices, sma, returns, volatility, upper_band, lower_band, window_size, ticker)

if __name__ == '__main__':
    main()
