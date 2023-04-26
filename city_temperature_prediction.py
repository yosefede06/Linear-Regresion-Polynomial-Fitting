from utils import split_train_test
from polynomial_fitting import PolynomialFitting

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

NOT_ISRAEL_COUNTRIES = ["Jordan", "South Africa", "The Netherlands"]


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df = df[df.notnull()]
    df = df[~df.duplicated()]
    df = df[df.Temp > 0]
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Year"] = df["Year"]
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/city_temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_df = df[df["Country"] == "Israel"]
    px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year").write_image("temperature/israel_daily_temperature.png")

    std_by_month = israel_df.groupby("Month", as_index=False)["Temp"].std()
    fig = px.bar(std_by_month, x="Month", y="Temp", title="Temperature Standard Deviation Over Years")
    fig.update_layout(title={"x": 0.5})

    fig.write_image("temperature/israel_month_temperature.png")
    # Question 3 - Exploring differences between countries
    grouped = df.groupby(["Country", "Month"], as_index=False).agg(mean=("Temp", "mean"), std=("Temp", "std"))

    fig = px.line(grouped, x="Month",
                  y="mean",
                  error_y="std",
                  color="Country",
                  title="Average Monthly Temperatures",
                  labels={"Month": "Month", "mean": "Mean Temperature"})

    fig.write_image("mean.temp.different.countries.png")

    # Question 4 - Fitting model for different values of `k`
    X_israel = israel_df['DayOfYear']
    y_israel = israel_df['Temp']
    train_X, train_y, test_X, test_y = split_train_test(X_israel, y_israel)
    all_k = np.arange(1, 11)
    loss = np.zeros(10)
    for i in range(1, 11):
        polynomial_fitting = PolynomialFitting(i)
        fitted = polynomial_fitting.fit(train_X, train_y)
        loss[i-1] = np.round(fitted.loss(test_X, test_y), 2)
    k_loss = pd.DataFrame({"k": all_k, "loss": loss})
    print(k_loss)
    px.bar(k_loss, x="k", y="loss", text="loss",
           title=r"$\text{Test Error For Different Values of }k$")\
        .write_image("temperature/k_pol.png")

    # Question 5 - Evaluating fitted model on different countries
    polynomial_fitting = PolynomialFitting(5)
    fitted = polynomial_fitting.fit(X_israel, y_israel)
    loss_arr = np.zeros(3)
    for i in range(3):
        loss_arr[i] = polynomial_fitting.loss(df[df.Country == NOT_ISRAEL_COUNTRIES[i]]["DayOfYear"],
                                              df[df.Country == NOT_ISRAEL_COUNTRIES[i]]["Temp"])
    countries_loss = pd.DataFrame({"Country": NOT_ISRAEL_COUNTRIES, "loss": np.round(loss_arr, 2)})
    print(countries_loss)
    px.bar(countries_loss,
           x="Country",
           y="loss",
           text="loss",
           color="Country",
           title="Loss over countries for model fitted over Israel").write_image("temperature/countries_loss.png")
