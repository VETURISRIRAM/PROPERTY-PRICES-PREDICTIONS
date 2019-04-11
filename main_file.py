"""
@author: Sriram Veturi
@purpose: Truss Case Study
@date: 04/02/2019
"""

import json
import xlrd
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def get_dataframe():
    """
    This function should load and return the dataframe
    :return: df (dataframe)
    """

    spaces_df = pd.read_excel('application/All_Test_Data_With_Descriptions.xlsx', sheetname='Available Spaces by Week')
    properties_df = pd.read_excel('application/All_Test_Data_With_Descriptions.xlsx', sheetname='Properties with Available Space')
    properties_df = properties_df.fillna(properties_df.median())
    df = pd.merge(spaces_df, properties_df, on='propertyId', how='inner')
    df = df.drop(columns=['marketName', 'city', 'state_id'])
    return df


def visualizations(df):
    """
    This function should plot the bar plots of all the columns
    :param df:  dataframe
    :return: None (Just Plotting)
    """
    df_columns = df.columns
    for col in df_columns:

        plt.figure(figsize=(7, 7))
        plt.tight_layout()
        p2 = sns.countplot(df[col])
        plt.title("{} Distribution".format(col))
        plt.show()


def plot_outliers(df):
    """
    This Fucntion shoudl plot the outliers in the dataset
    :param df: dataframe
    :return: None (Just Plotting)
    """

    df_columns = df.columns
    for col in df_columns:
        sns.boxplot(x=df[col])
        plt.show()

def preprocess_dataframe(df):
    """
    This function should preprocess the dataframe
    :param df: dataframe
    :return: df (Preprocessed DataFrame)
    """

    # Preprocess week ending column
    week_ending = df['weekEnding']
    month = [int(str(date)[5:7]) for date in list(week_ending)]
    df['month'] = pd.Series(month)
    df = df.drop(columns=['weekEnding'])
    # Preprocess postal-code column
    postal_codes = df['postal_code']
    postal_prefix = [str(code)[:5] for code in list(postal_codes)]
    df = df.drop(columns=['postal_code'])
    df['postal_code'] = pd.Series(postal_prefix)
    # Preprocess categorical columns
    catData = df.select_dtypes(include=['object'])
    catColumns = catData.columns
    df = df.drop(columns=catColumns)
    mapped_data = dict()
    for x in catData.columns:

        uniqueValues = catData[x].unique()
        mapping = dict(zip(uniqueValues, np.arange(float(len(uniqueValues)))))
        mapped_data[x] = mapping
        catData[x] = catData[x].map(mapping)

    df = pd.concat([df, catData], axis=1)
    df_columns = list(df.columns)
    for col in df_columns:

        df.plot.scatter(x=col, y='priceSF')
        plt.show()
    visualizations(df)
    plot_outliers(df)
    scaler = StandardScaler()
    scaledData = scaler.fit_transform(df)
    df = pd.DataFrame(data=scaledData, columns=df_columns)
    class_variable = df['priceSF']
    df = df.drop(columns=['priceSF'])
    df['priceSF'] = class_variable

    return df, mapped_data


def correlationFigure(featureVariablesMain, targetVariable):
    """
    This fucntion should plot the correlations plot
    :param featureVariablesMain: The entire dataframe
    :param targetVariable: Class Label 'priceSF'
    :return: correlations (correlation coefficients wrt class label)
    """

    # Calculate correlation
    def correlationCalculation(targetVariable, featureVariables, features):
        """
        This function should calculate the correlation coefficients.
        :param targetVariable: Class Label 'priceSF'.
        :param featureVariables: The features variables.
        :param features: column names of the features.
        :return:
        """

        columns = [] # For maintaining the feature names
        values = [] # For maintaining the corr values of features with "priceSF"
        # Traverse through all the input features
        for x in features:

            if x is not None:

                columns.append(x) # Append the column name
                # Calculate the correlation
                c = np.corrcoef(featureVariables[x], featureVariables[targetVariable])
                absC = abs(c) # Absolute value because important values might miss
                values.append(absC[0,1])

        dataDict = {'features': columns, 'correlation_values': values}
        corrValues = pd.DataFrame(dataDict)
        # Sort the value by correlation values
        sortedCorrValues = corrValues.sort_values(by="correlation_values")
        # Plot the graph to show the features with their correlation values
        figure, ax = plt.subplots(figsize=(15, 45), squeeze=True)
        ax.set_title("Correlation Coefficients of Features")
        sns.barplot(x=sortedCorrValues.correlation_values, y=sortedCorrValues['features'], ax=ax)
        ax.set_ylabel("-----------Corr Coefficients--------->")
        plt.show()
        return sortedCorrValues


    # Make a list of columns
    columns = []
    for x in featureVariablesMain.columns:

        columns.append(x)
    # Remove "priceSF" from df
    columns.remove(targetVariable)
    # Compute correlations
    correlations = correlationCalculation(targetVariable, featureVariablesMain, columns)
    return correlations


def plot_correlations(dataset):
    """
    Function to plot the correlations
    :param dataset: dataframe
    :return: importantFeatures (Top correlating features)
    """

    target = "priceSF"
    targetVariable = dataset['priceSF'].to_frame()
    corrData = correlationFigure(dataset, target)
    importantFeatures = corrData.sort_values(by="correlation_values", ascending=True).tail(5)
    return importantFeatures


def split_dataset(dataset):
    """
    This fucntion should split the dataset in to train and test sets
    :param dataset: dataframe
    :return: X_train, X_test, y_train, y_test (Split sets)
    """

    X = dataset.iloc[:, 2:3].values
    y = dataset.iloc[:, -1].values
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
    return X_train, X_test, y_train, y_test


def simple_linear_regression(X_train, X_test, y_train, y_test):
    """
    This fucntion should perform simple linear regression
    :param X_train: Train Features
    :param X_test: Test Features
    :param y_train: Train Classes
    :param y_test: Test Classes
    :return: y_pred (Predicted values)
    """

    # Fitting Simple Linear Regression to the Training set
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    # Visualising the Training set results
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title('Rentable Area Space vs Price (Training set)')
    plt.xlabel('Price')
    plt.ylabel('Rentable Area Space')
    plt.show()
    # Visualising the Test set results
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title('Rentable Area Space vs Price (Test set)')
    plt.xlabel('Price')
    plt.ylabel('Rentable Area Space')
    plt.show()
    return y_pred


def polynomial_regression(X_train, X_test, y_train, y_test):
    """
    This function should perform polynomial regression
    :param X_train: Train Features
    :param X_test: Test Features
    :param y_train: Train Classes
    :param y_test: Test Classes
    :return: y_pred (Predicted values)
    """

    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X_train)
    poly_reg.fit(X_poly, y_train)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y_train)
    # Visualising the Polynomial Regression results (for higher resolution and smoother curve)
    X_grid = np.arange(min(X_train), max(X_train), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
    plt.title('Rentable Area Space vs Price (Polynomial Regression) Training Set')
    plt.xlabel('Price')
    plt.ylabel('Rentable Area Space')
    plt.show()
    # Visualising the Polynomial Regression results (for higher resolution and smoother curve)
    X_grid = np.arange(min(X_test), max(X_test), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
    plt.title('Rentable Area Space vs Price (Polynomial Regression) Testing Set')
    plt.xlabel('Price')
    plt.ylabel('Rentable Area Space')
    plt.show()
    y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))
    return y_pred


def support_vector_regression(X_train, X_test, y_train, y_test):
    """
    This function should perform support vector regression
    :param X_train: Train Features
    :param X_test: Test Features
    :param y_train: Train Classes
    :param y_test: Test Classes
    :return: y_pred (Predicted values)
    """

    # Fitting SVR to the dataset
    regressor = SVR(kernel='rbf')
    regressor.fit(X_train, y_train)
    # Predicting a new result
    y_pred = regressor.predict(X_test)
    # Visualising the SVR results (for higher resolution and smoother curve)
    X_grid = np.arange(min(X_train), max(X_train), 0.01)  # choice of 0.01 instead of 0.1 step because the data is feature scaled
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_grid, regressor.predict(X_grid), color='blue')
    plt.title('Rentable Area Space vs Price (SVR) Training Set')
    plt.xlabel('Price')
    plt.ylabel('Rentable Area Space')
    plt.show()
    # Visualising the SVR results (for higher resolution and smoother curve)
    X_grid = np.arange(min(X_test), max(X_test),0.01)  # choice of 0.01 instead of 0.1 step because the data is feature scaled
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_grid, regressor.predict(X_grid), color='blue')
    plt.title('Rentable Area Space vs Price (SVR) Testing Set')
    plt.xlabel('Price')
    plt.ylabel('Rentable Area Space')
    plt.show()
    return y_pred


def decision_tree_regression(X_train, X_test, y_train, y_test):
    """
    This function should perform decision tree regression
    :param X_train: Train Features
    :param X_test: Test Features
    :param y_train: Train Classes
    :param y_test: Test Classes
    :return: y_pred (Predicted values)
    """

    # Fitting Decision Tree Regression to the dataset
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(X_train, y_train)
    # Predicting a new result
    y_pred = regressor.predict(X_test)
    # Visualising the Decision Tree Regression results (higher resolution)
    X_grid = np.arange(min(X_train), max(X_train), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_grid, regressor.predict(X_grid), color='blue')
    plt.title('Rentable Area Space vs Price (Decision Tree Regression)')
    plt.xlabel('Price')
    plt.ylabel('Rentable Area Space')
    plt.show()
    # Visualising the Decision Tree Regression results (higher resolution)
    X_grid = np.arange(min(X_test), max(X_test), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_grid, regressor.predict(X_grid), color='blue')
    plt.title('Rentable Area Space vs Price (Decision Tree Regression) Testing Set')
    plt.xlabel('Price')
    plt.ylabel('Rentable Area Space')
    plt.show()
    return y_pred


def random_forest_regression(X_train, X_test, y_train, y_test):
    """
    This function should perform random forest regression
    :param X_train: Train Features
    :param X_test: Test Features
    :param y_train: Train Classes
    :param y_test: Test Classes
    :return: y_pred (Predicted values)
    """

    # Fitting Random Forest Regression to the dataset
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(X_train, y_train)
    # Predicting a new result
    y_pred = regressor.predict(X_test)
    # Visualising the Random Forest Regression results (higher resolution)
    X_grid = np.arange(min(X_train), max(X_train), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_grid, regressor.predict(X_grid), color='blue')
    plt.title('Rentable Area Space vs Price (Random Forest Regression) Training Set')
    plt.xlabel('Price')
    plt.ylabel('Rentable Area Space')
    plt.show()
    # Visualising the Random Forest Regression results (higher resolution)
    X_grid = np.arange(min(X_test), max(X_test), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_grid, regressor.predict(X_grid), color='blue')
    plt.title('Rentable Area Space vs Price (Random Forest Regression) Testing Set')
    plt.xlabel('Price')
    plt.ylabel('Rentable Area Space')
    plt.show()
    return y_pred


def mean_absolute_error(y_test, y_pred):
    """
    This fucntion should return evaluation results
    :param y_test: Actual Set
    :param y_pred: Predicted Set
    :return: Evaluations (mae, mse, mpe, mape, rse)
    """

    mae_sum = 0
    mse_sum = 0
    mape_sum = 0
    mpe_sum = 0
    for y_actual, y_prediction in zip(y_test, y_pred):

        mae_sum += abs(y_actual - y_prediction)
        mse_sum += (y_actual - y_prediction)**2
        mape_sum += (abs((y_actual - y_prediction)) / y_actual)
        mpe_sum += ((y_actual - y_prediction) / y_actual)

    mae = mae_sum / len(y_test)
    mse = mse_sum / len(y_test)
    mape = mape_sum / len(y_test)
    mpe = mpe_sum / len(y_test)
    rse = r2_score(y_true=y_test, y_pred=y_pred)
    return mae, mse, mape, mpe, rse


def model_simple_linear_regression(X_train, X_test, y_train, y_test):
    """
    This function is a driving fucntion for simple linear regression
    :param X_train: Train Features
    :param X_test: Test Features
    :param y_train: Train Classes
    :param y_test: Test Classes
    :return: None (Just Printing stuff)
    """

    y_pred = simple_linear_regression(X_train, X_test, y_train, y_test)
    mae, mse, mape, mpe, rse = mean_absolute_error(y_test, y_pred)
    print("\nSimple Linear Regreesion Model Evaluations:\n")
    print("Mean Absolute Error: ", mae)
    print("Mean Squared Error: ", mse)
    print("Mean Absolute Percentage Error: ", mape)
    print("Mean Percentage Error: ", mpe)
    print("R Squared Error: ", rse)


def model_polynomial_regression(X_train, X_test, y_train, y_test):
    """
    This function is a driving fucntion for polynomial regression
    :param X_train: Train Features
    :param X_test: Test Features
    :param y_train: Train Classes
    :param y_test: Test Classes
    :return: None (Just Printing stuff)
    """

    y_pred = polynomial_regression(X_train, X_test, y_train, y_test)
    mae, mse, mape, mpe, rse = mean_absolute_error(y_test, y_pred)
    print("\nPolynomial Regreesion Model Evaluations:\n")
    print("Mean Absolute Error: ", mae)
    print("Mean Squared Error: ", mse)
    print("Mean Absolute Percentage Error: ", mape)
    print("Mean Percentage Error: ", mpe)
    print("R Squared Error: ", rse)


def model_svr_regression(X_train, X_test, y_train, y_test):
    """
    This function is a driving fucntion for support vector regression
    :param X_train: Train Features
    :param X_test: Test Features
    :param y_train: Train Classes
    :param y_test: Test Classes
    :return: None (Just Printing stuff)
    """

    y_pred = support_vector_regression(X_train, X_test, y_train, y_test)
    mae, mse, mape, mpe, rse = mean_absolute_error(y_test, y_pred)
    print("\nPolynomial Regreesion Model Evaluations:\n")
    print("Mean Absolute Error: ", mae)
    print("Mean Squared Error: ", mse)
    print("Mean Absolute Percentage Error: ", mape)
    print("Mean Percentage Error: ", mpe)
    print("R Squared Error: ", rse)


def model_decision_tree_regression(X_train, X_test, y_train, y_test):
    """
    This function is a driving fucntion for decision tree regression
    :param X_train: Train Features
    :param X_test: Test Features
    :param y_train: Train Classes
    :param y_test: Test Classes
    :return: None (Just Printing stuff)
    """

    y_pred = decision_tree_regression(X_train, X_test, y_train, y_test)
    mae, mse, mape, mpe, rse = mean_absolute_error(y_test, y_pred)
    print("\nPolynomial Regreesion Model Evaluations:\n")
    print("Mean Absolute Error: ", mae)
    print("Mean Squared Error: ", mse)
    print("Mean Absolute Percentage Error: ", mape)
    print("Mean Percentage Error: ", mpe)
    print("R Squared Error: ", rse)


def model_random_forest_regression(X_train, X_test, y_train, y_test):
    """
    This function is a driving fucntion for random forest regression
    :param X_train: Train Features
    :param X_test: Test Features
    :param y_train: Train Classes
    :param y_test: Test Classes
    :return: None (Just Printing stuff)
    """

    y_pred = random_forest_regression(X_train, X_test, y_train, y_test)
    mae, mse, mape, mpe, rse = mean_absolute_error(y_test, y_pred)
    print("\nRandom Forest Regreesion Model Evaluations:\n")
    print("Mean Absolute Error: ", mae)
    print("Mean Squared Error: ", mse)
    print("Mean Absolute Percentage Error: ", mape)
    print("Mean Percentage Error: ", mpe)
    print("R Squared Error: ", rse)


# Main function starts here..
if __name__ == "__main__":

    # Get dataset
    df = get_dataframe()
    # Preprocess dataset
    df, mappings = preprocess_dataframe(df)
    print(json.dumps(mappings, indent=1))
    # Get Important features
    importantFeatures = plot_correlations(df)
    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(df)
    # Apply Simple Linear Regression
    model_simple_linear_regression(X_train, X_test, y_train, y_test)
    # Apply Polynomial Regression
    model_polynomial_regression(X_train, X_test, y_train, y_test)
    # Apply Support Vector Regression
    model_svr_regression(X_train, X_test, y_train, y_test)
    # Apply Decision Tree Regression
    model_decision_tree_regression(X_train, X_test, y_train, y_test)
    # Apply Random Forest Regression
    model_random_forest_regression(X_train, X_test, y_train, y_test)
