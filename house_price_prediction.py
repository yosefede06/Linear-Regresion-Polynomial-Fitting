import sklearn.model_selection

from utils import split_train_test
from utils import split_train_test
from linear_regression import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

AVERAGE_COLUMNS = ["sqft_living", "sqft_lot", "sqft_above", "bathrooms", "floors", "sqft_basement", "bedrooms"]
AVERAGE_COLUMNS_NO_DECIMALS = ["waterfront", "view", "condition", "grade"]
ALL_AVERAGE_COLUMNS = AVERAGE_COLUMNS + AVERAGE_COLUMNS_NO_DECIMALS

def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # Remove rows which price == null and price is negative
    # clean X rows
    X = X[y.notnull()]
    # clean Y rows
    y = y[y.notnull()]
    # remove rows with no id's
    y = y[X["id"].notnull() & X["id"] > 0 ]
    # remove rows with no id's
    X = X[X["id"].notnull() & X["id"] > 0]
    id_column = X["id"]
    y = y[~id_column.duplicated()]
    X = X[~id_column.duplicated()]

    X = X.drop(columns=["id", "lat", "long", "date", "sqft_lot15", "sqft_living15"])
    X = pd.get_dummies(X, prefix="zip", columns=['zipcode'])
    X["five_years_period_built"] = (X["yr_built"] / 5).apply(np.ceil)
    X = pd.get_dummies(X, prefix='five_years_period_built_',  columns=['five_years_period_built'])
    X = X.drop(columns="yr_built")
    ren_col = X["yr_renovated"]
    X["recently_renovated"] = np.where(ren_col >= np.percentile(ren_col.unique(), 80), 1, 0)
    X = X.drop(columns="yr_renovated")
    y = y[X["bedrooms"] < 20]
    X = X[X["bedrooms"] < 20]
    y = y[X["bathrooms"] < 20]
    X = X[X["bathrooms"] < 20]
    y = y[X["sqft_lot"] < 1200000]
    X = X[X["sqft_lot"] < 1200000]
    return X, y

def preprocess_data_train(X: pd.DataFrame, y: Optional[pd.Series] = None):
    X = X[y > 0]
    # clean Y rows
    y = y[y > 0]
    #
    for feauture in ["sqft_living", "sqft_lot", "sqft_above"]:
        y = y[X[feauture] > 0]
        X = X[X[feauture] > 0]

    for feauture in ["bathrooms", "floors", "sqft_basement", "bedrooms"]:
        y = y[X[feauture] >= 0]
        X = X[X[feauture] >= 0]

    y = y[X["waterfront"].isin([0, 1]) &
            X["view"].isin(range(5)) &
            X["condition"].isin(range(1, 6)) &
            X["grade"].isin(range(1, 15))]

    X = X[X["waterfront"].isin([0, 1]) &
          X["view"].isin(range(5)) &
          X["condition"].isin(range(1, 6)) &
          X["grade"].isin(range(1, 15))]

    return X, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    X = X.filter(regex='^(?!zip_|five_years_period_built).*')
    for f in X:
        rho = np.cov(X[f], y)[0, 1] / (np.std(X[f]) * np.std(y))

        fig = px.scatter(pd.DataFrame({'x': X[f], 'y': y}), x="x", y="y",
                         trendline="ols",
                         title=f"correlation between {f} values and Response <br>Pearson Correlation {rho}",
                         labels={"x": f"{f} Values", "y": "Response Values"})
        fig.write_image(output_path + f"/pearson/pearson.correlation.{f}.png")


def replace_non_legal_value_for_test(X: pd.DataFrame, train_average_vector, train_cols) -> np.ndarray:
    test_columns = X.columns
    for col in train_cols:
        if col not in test_columns:
            X[col] = 0
    for feature in ALL_AVERAGE_COLUMNS:
        X[feature] = X[feature].mask(X[feature].isnull(), train_average_vector[feature])
    for feature in ["sqft_living", "sqft_lot", "sqft_above"]:
        X[feature] = X[feature].mask(X[feature] <= 0, train_average_vector[feature])
    for feature in ["bathrooms", "floors", "sqft_basement", "bedrooms"]:
        X[feature] = X[feature].mask(X[feature] < 0, train_average_vector[feature])
    grade = ~X["grade"].isin(range(1, 15))
    waterfront = ~X["waterfront"].isin([0, 1])
    view = ~X["view"].isin(range(5))
    condition = ~X["condition"].isin(range(1, 6))
    X['grade'] = X['grade'].mask(grade, train_average_vector["grade"])
    X['waterfront'] = X['waterfront'].mask(waterfront, train_average_vector["waterfront"])
    X['view'] = X['view'].mask(view, train_average_vector["view"])
    X['condition'] = X['condition'].mask(condition, train_average_vector["condition"])
    return X


def train_average_features(X: pd.DataFrame):
    av_X = X[AVERAGE_COLUMNS].mean(axis=0).round(2)
    av_X_1 = X[AVERAGE_COLUMNS_NO_DECIMALS].mean(axis=0).round(0)
    return pd.concat([av_X, av_X_1])


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv("../datasets/house_prices.csv")
    np.random.seed(0)
    X = df.drop(columns=['price'])
    y = df['price']

    # Question 1 - split data into train and test sets
    train_X, train_y, test_X, test_y = split_train_test(X=X, y=y)

    # Question 2 - Preprocessing of housing prices dataset

    train_X, train_y = preprocess_data(train_X, train_y)
    test_X, test_y = preprocess_data(test_X, test_y)

    train_X, train_y = preprocess_data_train(train_X, train_y)
    train_average_vector = train_average_features(train_X)
    # print(test_X.columns)
    test_X = replace_non_legal_value_for_test(test_X, train_average_vector, train_X.columns)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:

    # 1) Sample p% of the overall training data
    TOP = 101
    avg_list = np.zeros(TOP - 10)
    std_list = np.zeros(TOP - 10)
    percent_list = np.arange(10, TOP)
    for percent in range(10, TOP):
    #   2) Fit linear model (including intercept) over sampled set
        predicted_loss_list = np.array([])
        for j in range(10):
            train_X_proportion = train_X.sample(frac=percent/100)
            train_y_proportion = train_y.loc[train_X_proportion.index]
            linear_reg = LinearRegression(True)
            linear_reg._fit(train_X_proportion, train_y_proportion)
            #   3) Test fitted model over test set
            predicted_loss_list = np.append(predicted_loss_list, linear_reg._loss(test_X, test_y))
        #   4) Store average and variance of loss over test set
        avg_list[percent - 10], std_list[percent - 10] = predicted_loss_list.mean(), predicted_loss_list.std()
        print("Percentage of Training Set: " + str(percent) + ", Mean: " + str(avg_list[percent - 10]) + ", STD: " + str(std_list[percent - 10]))
    fig = go.Figure([go.Scatter(x=percent_list, y=avg_list - 2 * std_list, fill=None, mode="lines", line=dict(color="red")),
                     go.Scatter(x=percent_list, y=avg_list + 2 * std_list, fill='tonexty', mode="lines", line=dict(color="red")),
                     go.Scatter(x=percent_list, y=avg_list, mode="markers+lines", marker=dict(color="black"))],
                    layout=go.Layout(title="Test MSE as Function Of Training Size",
                                     xaxis=dict(title="Percentage of Training Set"),
                                     yaxis=dict(title="MSE Over Test Set"),
                                     showlegend=False))
    fig.write_image("mse.over.training.percentage.png")

    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)


