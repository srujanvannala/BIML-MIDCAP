import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="marico Equity Analysis", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('marico.csv')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE')
    return df

df = load_data()

st.title("Marico Equity Analysis Dashboard")

# Sidebar for navigation
option = st.sidebar.selectbox(
    'Select Analysis Type',
    ['Descriptive Analysis', 'Diagnostic Analysis', 'Predictive Analysis', 'Prescriptive Analysis']
)

# ---------------- Descriptive Analysis ----------------
if option == 'Descriptive Analysis':
    st.header("Descriptive Analysis")
    st.write("Summary statistics of marico stock data")

    # Show raw data toggle
    if st.checkbox("Show raw data"):
        st.dataframe(df)

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("Price Over Time")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['DATE'], df['CLOSE'], label='Close Price', color='blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (INR)')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Volume Over Time")
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.bar(df['DATE'], df['VOLUME'], color='orange')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    st.pyplot(fig2)

# ---------------- Diagnostic Analysis ----------------
elif option == 'Diagnostic Analysis':
    st.header("Diagnostic Analysis")
    st.write("Investigate trends and relationships")

    st.subheader("Correlation Matrix")
    corr = df[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Moving Averages")
    window = st.slider("Select moving average window (days)", min_value=3, max_value=30, value=10)
    df['MA'] = df['CLOSE'].rolling(window=window).mean()

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['DATE'], df['CLOSE'], label='Close Price')
    ax.plot(df['DATE'], df['MA'], label=f'{window}-day MA')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Price Volatility")
    df['RETURNS'] = df['CLOSE'].pct_change()
    df['VOLATILITY'] = df['RETURNS'].rolling(window=window).std() * np.sqrt(window)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df['DATE'], df['VOLATILITY'], color='red')
    ax.set_title('Rolling Volatility')
    st.pyplot(fig)

# ---------------- Predictive Analysis ----------------
elif option == 'Predictive Analysis':
    st.header("Predictive Analysis")
    st.write("Forecasting future marico stock closing price using ARIMA")

    # Prepare data
    ts = df.set_index('DATE')['CLOSE']

    st.subheader("ARIMA Parameters")
    p = st.number_input("AR (p) parameter", min_value=0, max_value=5, value=2)
    d = st.number_input("Degree of differencing (d)", min_value=0, max_value=2, value=1)
    q = st.number_input("MA (q) parameter", min_value=0, max_value=5, value=2)

    if st.button("Run Forecast"):
        try:
            model = ARIMA(ts, order=(p,d,q))
            model_fit = model.fit()

            # Forecast next 15 days
            forecast = model_fit.forecast(steps=15)
            forecast_dates = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=15)

            forecast_df = pd.DataFrame({'DATE': forecast_dates, 'FORECAST': forecast})
            st.write(forecast_df)

            # Plot forecast
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(ts.index, ts, label='Historical Close Price')
            ax.plot(forecast_df['DATE'], forecast_df['FORECAST'], label='Forecast', color='green')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------- Prescriptive Analysis ----------------
elif option == 'Prescriptive Analysis':
    st.header("Prescriptive Analysis")
    st.write("Suggest trading actions based on moving average crossover strategy")

    short_window = st.slider("Short Moving Average Window", min_value=3, max_value=15, value=5)
    long_window = st.slider("Long Moving Average Window", min_value=10, max_value=30, value=20)

    df['SHORT_MA'] = df['CLOSE'].rolling(window=short_window).mean()
    df['LONG_MA'] = df['CLOSE'].rolling(window=long_window).mean()

    # Generate signals
    df['SIGNAL'] = 0
    df['SIGNAL'][long_window:] = np.where(df['SHORT_MA'][long_window:] > df['LONG_MA'][long_window:], 1, 0)
    df['POSITION'] = df['SIGNAL'].diff()

    st.subheader("Moving Average Crossover Chart")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['DATE'], df['CLOSE'], label='Close Price', alpha=0.5)
    ax.plot(df['DATE'], df['SHORT_MA'], label=f'Short MA ({short_window} days)')
    ax.plot(df['DATE'], df['LONG_MA'], label=f'Long MA ({long_window} days)')

    # Buy signals
    ax.plot(df.loc[df['POSITION'] == 1, 'DATE'],
            df.loc[df['POSITION'] == 1, 'CLOSE'],
            '^', markersize=12, color='g', label='Buy Signal')

    # Sell signals
    ax.plot(df.loc[df['POSITION'] == -1, 'DATE'],
            df.loc[df['POSITION'] == -1, 'CLOSE'],
            'v', markersize=12, color='r', label='Sell Signal')

    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    ### Trading Strategy Explanation
    - **Buy** when the short-term moving average crosses above the long-term moving average.
    - **Sell** when the short-term moving average crosses below the long-term moving average.
    """)

